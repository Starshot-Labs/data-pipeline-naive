// The one seam between the pipeline and the volumes.
//
// Every operation that needs a volume mounted goes through here, served one of two ways:
//
//   remote  inside `dc-scene-ops` over HTTP. Always used from a laptop, and always used for
//           anything touching the farm's volumes.
//   direct  `ops.mjs` in-process, when the pipeline is itself on Modal with `/scene` mounted
//           (`SCENE_OPS_DIRECT=1`, set by `modal/pipeline.py`). No round trip, and no voxel
//           grid serialised out and back.
//
// The split is not a preference, it is a constraint. Modal volumes are snapshots: a container
// must `reload()` to see another's commits, and only Python can call that. So a command that
// reads a farm volume — which the farm writes, on its own schedule — has to run where a
// reload can happen around it. `/scene` is different: during a run the pipeline container is
// its only writer, so it already sees its own work, and Python reloads and commits at stage
// boundaries where the ownership changes hands.
//
// Meshes never travel either way.

import fs from 'node:fs';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const run = promisify(execFile);
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const DEFAULT_BASE_URL = 'https://starshot-aitools--dc-scene-ops-web.modal.run';
const LANE_PARAM = { tex: 'texture_dir', geo: 'geometry_dir' };

// Read lazily: this module is evaluated before an entry point loads .env.
const isDirect = () => process.env.SCENE_OPS_DIRECT === '1';
const baseUrl = () => process.env.SCENE_BASE_URL ?? DEFAULT_BASE_URL;
const timeoutMs = () => Number(process.env.SCENE_REQUEST_TIMEOUT_S ?? 900) * 1000;
const inputVolume = () => process.env.TRELLIS_INPUT_VOLUME ?? 't2farm-input-v2';
const modalBin = () => process.env.TRELLIS_MODAL_BIN ?? 'modal';
const stageDir = () => path.resolve(ROOT, process.env.TRELLIS_STAGE ?? '.trellis-stage');

// These read a volume the farm writes, so they need a reload around them that only the
// service can do. They stay remote even when the rest of the pipeline is running on Modal.
const REMOTE_ONLY = new Set(['stage', 'assets', 'collect']);

// Imported on first use so a remote-mode run never pays to load sharp, fzstd and three.
let commands = null;
const inProcess = async () => (commands ??= (await import('./ops.mjs')).COMMANDS);

async function call(command, payload = {}) {
  if (isDirect() && !REMOTE_ONLY.has(command)) return (await inProcess())[command](payload);

  const response = await fetch(`${baseUrl()}/${command}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(timeoutMs()),
  });
  if (!response.ok) {
    throw new Error(`scene-ops ${command} ${response.status}: ${(await response.text()).slice(0, 300)}`);
  }
  return response.json();
}

export const url = () => (isDirect() ? `${baseUrl()} + mounted /scene` : baseUrl());

export async function health() {
  try {
    const response = await fetch(`${baseUrl()}/health`, { signal: AbortSignal.timeout(60_000) });
    return response.ok ? 'healthy' : `HTTP ${response.status}`;
  } catch (err) {
    return err.message;
  }
}

/**
 * Hard-links the images into a per-lane tree and pushes each lane with the Modal CLI, which
 * is the only route onto a volume from outside one. Linking costs neither disk nor time; the
 * copy is reached only when the stage lands on a different filesystem, which a link cannot
 * span.
 */
async function putWithCli(runId, objects) {
  const stage = stageDir();
  fs.rmSync(stage, { recursive: true, force: true });
  try {
    const lanes = new Set();
    for (const { source, key, lane } of objects) {
      const directory = path.join(stage, lane);
      fs.mkdirSync(directory, { recursive: true });
      const destination = path.join(directory, `${key}${path.extname(source)}`);
      try {
        fs.linkSync(source, destination);
      } catch {
        fs.copyFileSync(source, destination);
      }
      lanes.add(lane);
    }
    // One call per lane at any size — the volume has no request-body or timeout ceiling.
    for (const lane of lanes) {
      await run(
        modalBin(),
        ['volume', 'put', inputVolume(), path.join(stage, lane), `uploads/${runId}-${lane}`, '--force'],
        { env: { ...process.env, PYTHONIOENCODING: 'utf-8' }, maxBuffer: 512 * 1024 * 1024 },
      );
    }
  } finally {
    fs.rmSync(stage, { recursive: true, force: true });
  }
}

/**
 * Puts the reference images where the farm's dispatcher will find them and returns the lane
 * directories `POST /run` wants. `objects` is `[{ source, key, lane }]`.
 *
 * On Modal the sources are already on `/scene`, so the service copies them volume-to-volume
 * and commits the farm's input for us. From a laptop they are on local disk, which no service
 * can see, so the only route is the Modal CLI.
 */
export async function stageImages(runId, objects) {
  const counts = {};
  for (const { lane } of objects) counts[lane] = (counts[lane] ?? 0) + 1;
  if (!Object.keys(counts).length) throw new Error('no images to stage');

  if (isDirect()) await call('stage', { run_id: runId, objects });
  else await putWithCli(runId, objects);

  const dirs = {};
  for (const [lane, param] of Object.entries(LANE_PARAM)) if (counts[lane]) dirs[param] = `uploads/${runId}-${lane}`;
  return { dirs, counts };
}

/** How many GLBs a campaign has written, which is what tells us it has finished. */
export const assetCount = (runId) => call('assets', { run_id: runId }).then((r) => r.count).catch(() => 0);

/**
 * Moves a finished campaign's meshes and their reference images off the farm's volumes onto
 * ours, named the way the sample names them. `objects` is one entry per mesh:
 * `{ sample, role, stem, key, lane }`.
 *
 * The image preferred is the centred crop Trellis reconstructed from, out of the `.st.zst`
 * shards under `<run>/images/`; the reply counts how many of those it found against how many
 * had to fall back to the photo we sent up.
 */
export const collect = (runId, objects) => call('collect', { run_id: runId, objects });

/** Occupancy grids for both of a sample's meshes: `objects` is `{ role, stem, resolution }`. */
export const voxelize = (sample, objects) => call('voxelize', { sample, objects });

/** Writes the posed meshes and their images into the published folder, and measures them. */
export const bake = (sample, objects) => call('bake', { sample, objects });

/**
 * The two small text files for many samples at once, written last so they mark each one
 * complete. `samples` is `[{ sample, files }]`.
 */
export const publish = (samples) => call('publish', { samples });

/** Sample ids already carrying a metadata.json on the volume. */
export const published = () => call('published').then((r) => new Set(r.ids));
