// Single-object image-to-3D on the hosted Trellis 2 HTTP API, mirroring
// benchmark-2/server/scripts/test_trellis.py: POST /generate spawns a job,
// GET /jobs/{id} is polled until it is done, GET /jobs/{id}/result is the GLB.
//
// One image in, one mesh out — nothing here knows about campaigns. The batched
// farm client that used to live at this path is now `trellis-farm.mjs`.

const DEFAULT_BASE_URL = 'https://starshot-aitools--starshot-assets-router-fastapi-app.modal.run';

// Read lazily: this module is evaluated before the entry point loads .env.
const baseUrl = () => process.env.TRELLIS_BASE_URL ?? DEFAULT_BASE_URL;
const pollMs = () => Number(process.env.TRELLIS_POLL_S ?? 2) * 1000;
const pollTimeoutMs = () => Number(process.env.TRELLIS_POLL_TIMEOUT_S ?? 600) * 1000;

// Per-request timeouts, same as the reference script: the spawn and each poll are
// small calls, the result download is a mesh of several MB.
const SPAWN_TIMEOUT_MS = 60_000;
const POLL_REQUEST_TIMEOUT_MS = 30_000;
const RESULT_TIMEOUT_MS = 180_000;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const detail = async (response, label) =>
  new Error(`${label} ${response.status}: ${(await response.text()).slice(0, 300)}`);

/** Cheap probe so a bad URL fails fast rather than on the first real job. */
export async function health() {
  const response = await fetch(`${baseUrl()}/health`, { signal: AbortSignal.timeout(10_000) });
  if (!response.ok) throw await detail(response, 'GET /health');
  return response.json();
}

/**
 * One image through the whole job lifecycle. Returns the GLB bytes.
 *
 * Spawn and download failures throw — there is nothing to salvage. A poll that errors
 * merely waits out the next interval, since dropping a job the GPU is still working on
 * over one flaky status call would pay for the mesh and then throw it away.
 */
export async function generateMesh({
  image,
  name = 'object.png',
  seed = Number(process.env.TRELLIS_SEED ?? 0),
  resolution = process.env.TRELLIS_RESOLUTION ?? '512',
  textureSize = Number(process.env.TRELLIS_TEXTURE_SIZE ?? 1024),
  decimationTarget = Number(process.env.TRELLIS_DECIMATION_TARGET ?? 500_000),
  log = () => {},
}) {
  const form = new FormData();
  form.append('image', new Blob([image], { type: 'image/png' }), name);
  form.append('seed', String(seed));
  form.append('resolution', String(resolution));
  form.append('texture_size', String(textureSize));
  form.append('decimation_target', String(decimationTarget));

  const started = Date.now();
  const spawn = await fetch(`${baseUrl()}/generate`, {
    method: 'POST',
    body: form,
    signal: AbortSignal.timeout(SPAWN_TIMEOUT_MS),
  });
  if (!spawn.ok) throw await detail(spawn, 'POST /generate');

  // A 200 whose body is not JSON is a router error page, not a job — say what came back.
  const spawnBody = await spawn.text();
  let jobId;
  try {
    jobId = JSON.parse(spawnBody).job_id;
  } catch {
    throw new Error(`POST /generate returned 200 but not JSON: ${spawnBody.slice(0, 300)}`);
  }
  if (!jobId) throw new Error(`POST /generate returned no job_id: ${spawnBody.slice(0, 300)}`);
  log(`trellis job ${jobId}`);

  const deadline = Date.now() + pollTimeoutMs();
  for (let polls = 1; ; polls++) {
    if (Date.now() >= deadline) throw new Error(`trellis job ${jobId} timed out after ${pollTimeoutMs() / 1000}s`);
    await sleep(pollMs());

    let status;
    try {
      const response = await fetch(`${baseUrl()}/jobs/${jobId}`, {
        signal: AbortSignal.timeout(POLL_REQUEST_TIMEOUT_MS),
      });
      if (!response.ok) throw await detail(response, `GET /jobs/${jobId}`);
      status = await response.json();
    } catch (err) {
      log(`poll failed, retrying: ${err.message}`);
      continue;
    }

    if (status.status === 'done') {
      log(`done after ${polls} poll(s), ${((Date.now() - started) / 1000).toFixed(1)}s`);
      break;
    }
    if (status.status === 'failed') {
      throw new Error(`trellis job ${jobId} failed: ${JSON.stringify(status.error ?? status).slice(0, 300)}`);
    }
    if (polls % 15 === 0) log(`still pending (${polls} polls, ${((Date.now() - started) / 1000).toFixed(0)}s)`);
  }

  const result = await fetch(`${baseUrl()}/jobs/${jobId}/result`, {
    signal: AbortSignal.timeout(RESULT_TIMEOUT_MS),
  });
  if (!result.ok) throw await detail(result, `GET /jobs/${jobId}/result`);

  const glb = Buffer.from(await result.arrayBuffer());
  if (glb.subarray(0, 4).toString('ascii') !== 'glTF') {
    throw new Error(`trellis job ${jobId} result is not a GLB (first bytes: ${glb.subarray(0, 8).toString('hex')})`);
  }
  return glb;
}
