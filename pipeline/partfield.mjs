// Client for `dc-partfield`, NVIDIA PartField as two calls:
//
//   segment  POST /segment  → build the part hierarchy, and take one cut of it
//   cut      POST /cut      → take another cut of one already built
//
// PartField predicts a feature field over a shape and clusters the per-triangle features under
// the mesh's own face adjacency. That clustering is a hierarchy rather than a segmentation, so
// a result is one level of it: a GLB of one named, coloured node per part, and a flat
// Int32Array in which entry i is the part face i of the uploaded mesh belongs to. The labels
// index the source mesh as it was sent, so they compose with anything else that reads it.
//
// How many parts a shape wants is not knowable up front — eight flatters a chair and shreds a
// mug — and the hierarchy already holds every answer, so `cut` returns another level of a
// finished job without a GPU or the model. It is seconds against the minute and a half the
// tree costs; `summary.levels` says which levels a job can answer.
//
// Both spawn and are polled the way the voxhammer and Trellis endpoints are.

const DEFAULT_BASE_URL = 'https://starshot-aitools--dc-partfield-web.modal.run';

// Read lazily: this module is evaluated before the entry point loads .env.
const baseUrl = () => process.env.PARTFIELD_BASE_URL ?? DEFAULT_BASE_URL;
const pollMs = () => Number(process.env.PARTFIELD_POLL_S ?? 10) * 1000;
const timeoutMs = () => Number(process.env.PARTFIELD_TIMEOUT_S ?? 3600) * 1000;

// Meshes run to tens of megabytes, so this deadline covers an upload, not a handshake.
const SPAWN_TIMEOUT_MS = 600_000;
const POLL_REQUEST_TIMEOUT_MS = 30_000;
const FILE_TIMEOUT_MS = 300_000;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const detail = async (response, label) =>
  new Error(`${label} ${response.status}: ${(await response.text()).slice(0, 300)}`);

export async function health() {
  const response = await fetch(`${baseUrl()}/health`, { signal: AbortSignal.timeout(30_000) });
  if (!response.ok) throw await detail(response, 'GET /health');
  return response.json();
}

/** Poll one job to completion, reporting each new stage. Returns the final status. */
async function awaitJob(jobId, { log }) {
  const deadline = Date.now() + timeoutMs();
  const started = Date.now();
  let stage = null;
  for (;;) {
    if (Date.now() >= deadline) throw new Error(`job ${jobId} timed out after ${timeoutMs() / 1000}s`);
    await sleep(pollMs());

    let status;
    try {
      const response = await fetch(`${baseUrl()}/jobs/${jobId}`, { signal: AbortSignal.timeout(POLL_REQUEST_TIMEOUT_MS) });
      if (!response.ok) throw await detail(response, `GET /jobs/${jobId}`);
      status = await response.json();
    } catch (err) {
      log(`poll failed, retrying: ${err.message}`);
      continue;
    }

    if (status.stage && status.stage !== stage) {
      stage = status.stage;
      log(`${stage} (${((Date.now() - started) / 60_000).toFixed(1)} min)`);
    }
    if (status.status === 'done') return status;
    if (status.status === 'failed') throw new Error(`job ${jobId} failed: ${String(status.error).slice(0, 500)}`);
  }
}

/** Download the named artifacts of a job as `{ name: bytes }`. */
async function download(jobId, names) {
  const files = {};
  for (const name of names ?? []) {
    const response = await fetch(`${baseUrl()}/jobs/${jobId}/file/${encodeURIComponent(name)}`, {
      signal: AbortSignal.timeout(FILE_TIMEOUT_MS),
    });
    if (!response.ok) throw await detail(response, `GET file ${name}`);
    files[name] = Buffer.from(await response.arrayBuffer());
  }
  return files;
}

/** `labels.bin` is little-endian int32, one per face, and Node runs nowhere else. */
const toLabels = (bytes) =>
  new Int32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));

/** A cut's three artifacts, taken by extension since they are named for the level they are. */
function unpack(files) {
  const of = (extension) => Object.entries(files).find(([name]) => name.endsWith(extension))?.[1];
  const summary = of('.json');
  if (!summary) throw new Error(`job returned no summary, only ${Object.keys(files).join(', ') || 'nothing'}`);
  return { glb: of('.glb'), labels: toLabels(of('.bin')), summary: JSON.parse(summary.toString('utf8')) };
}

/**
 * Build a mesh's part hierarchy and take one cut of it.
 *
 * `parts` is the cut to take; `maxClusters` is how deep the tree goes, and bounds what `cut`
 * can be asked for later. `option` only matters for a mesh that is not one connected
 * component, where it chooses how the pieces are bridged before clustering: 0 chains them,
 * 2 spans them along a minimum spanning tree.
 */
export async function segment({
  id = '',
  model,
  filename = 'mesh.glb',
  parts = 8,
  maxClusters = 20,
  option,
  withKnn,
  pointsPerFace,
  log = () => {},
}) {
  const form = new FormData();
  form.append('model', new Blob([model], { type: 'model/gltf-binary' }), filename);
  form.append('sample', id);
  form.append('num_parts', String(parts));
  form.append('max_clusters', String(maxClusters));
  if (option !== undefined) form.append('option', String(option));
  if (withKnn !== undefined) form.append('with_knn', String(withKnn));
  if (pointsPerFace !== undefined) form.append('n_point_per_face', String(pointsPerFace));

  const spawn = await fetch(`${baseUrl()}/segment`, { method: 'POST', body: form, signal: AbortSignal.timeout(SPAWN_TIMEOUT_MS) });
  if (!spawn.ok) throw await detail(spawn, 'POST /segment');
  const jobId = (await spawn.json()).job_id;
  if (!jobId) throw new Error('POST /segment returned no job_id');
  log(`segment job ${jobId}`);

  const status = await awaitJob(jobId, { log });
  const result = { jobId, ...unpack(await download(jobId, status.files)) };
  log(`${result.summary.num_parts} parts over ${result.summary.faces} faces in ${status.record?.total_seconds}s`);
  return result;
}

/**
 * Take another cut of a hierarchy `segment` already built, at a different number of parts.
 *
 * No GPU and no model — the tree is on the service, so this is a rebuild of the GLB against a
 * label array it already holds. `parts` has to be one of `summary.levels`.
 */
export async function cut({ jobId, parts, log = () => {} }) {
  const form = new FormData();
  form.append('job_id', jobId);
  form.append('num_parts', String(parts));

  const spawn = await fetch(`${baseUrl()}/cut`, { method: 'POST', body: form, signal: AbortSignal.timeout(SPAWN_TIMEOUT_MS) });
  if (!spawn.ok) throw await detail(spawn, 'POST /cut');

  const status = await awaitJob(jobId, { log });
  const result = { jobId, ...unpack(await download(jobId, status.files)) };
  log(`re-cut to ${result.summary.num_parts} parts in ${status.record?.total_seconds}s`);
  return result;
}

/** Free a job's features and hierarchy, which are tens of megabytes each. */
export async function discard(jobId) {
  const response = await fetch(`${baseUrl()}/jobs/${jobId}`, { method: 'DELETE', signal: AbortSignal.timeout(60_000) });
  if (!response.ok) throw await detail(response, `DELETE /jobs/${jobId}`);
  return response.json();
}
