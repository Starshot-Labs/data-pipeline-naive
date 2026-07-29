// Client for `dc-voxhammer`, the GPU half of experiment 2, split to match the service:
//
//   renderViews  POST /render      → the anchor view + 2D mask to insert the object into
//   (caller)     make 2d_edit.png  → done in the pipeline with nano-banana, off-GPU
//   editSample   POST /edit        → the 150-view render, features, voxel mask, and edit
//
// Splitting the round trip is the whole point: the inpaint that VoxHammer runs with FLUX
// (text-only, never sees the object) becomes a reference-image edit the caller controls,
// while the service keeps VoxHammer's method unchanged. Both calls are long jobs, so each
// spawns and is polled the same way the single-object Trellis endpoint is.

const DEFAULT_BASE_URL = 'https://starshot-aitools--dc-voxhammer-web.modal.run';

// Read lazily: this module is evaluated before the entry point loads .env.
const baseUrl = () => process.env.VOXHAMMER_BASE_URL ?? DEFAULT_BASE_URL;
const pollMs = () => Number(process.env.VOXHAMMER_POLL_S ?? 15) * 1000;
const renderTimeoutMs = () => Number(process.env.VOXHAMMER_RENDER_TIMEOUT_S ?? 1800) * 1000;
const editTimeoutMs = () => Number(process.env.VOXHAMMER_TIMEOUT_S ?? 7200) * 1000;

const SPAWN_TIMEOUT_MS = 120_000;
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
async function awaitJob(jobId, { timeoutMs, log }) {
  const deadline = Date.now() + timeoutMs;
  const started = Date.now();
  let stage = null;
  for (let polls = 1; ; polls++) {
    if (Date.now() >= deadline) throw new Error(`job ${jobId} timed out after ${timeoutMs / 1000}s`);
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

/**
 * Step 3. Uploads the two meshes and gets back the chosen anchor view and its 2D mask.
 * The job id it returns is what `editSample` addresses once the edited view exists.
 */
export async function renderViews({ id, model, mask, log = () => {} }) {
  const form = new FormData();
  form.append('model', new Blob([model], { type: 'model/gltf-binary' }), 'model.glb');
  form.append('mask', new Blob([mask], { type: 'model/gltf-binary' }), 'mask.glb');
  form.append('sample', id);

  const spawn = await fetch(`${baseUrl()}/render`, { method: 'POST', body: form, signal: AbortSignal.timeout(SPAWN_TIMEOUT_MS) });
  if (!spawn.ok) throw await detail(spawn, 'POST /render');
  const jobId = (await spawn.json()).job_id;
  if (!jobId) throw new Error('POST /render returned no job_id');
  log(`render job ${jobId}`);

  const status = await awaitJob(jobId, { timeoutMs: renderTimeoutMs(), log });
  const files = await download(jobId, status.files);
  return { jobId, view: status.view, render: files['images/2d_render.png'], mask: files['images/2d_mask.png'] };
}

/**
 * Steps 5-6. Hands the edited view to the job `renderViews` prepared, waits out the edit,
 * and returns every artifact it produced plus what the service recorded.
 */
export async function editSample({ jobId, edit, log = () => {} }) {
  const form = new FormData();
  form.append('job_id', jobId);
  form.append('edit_image', new Blob([edit], { type: 'image/png' }), '2d_edit.png');

  const spawn = await fetch(`${baseUrl()}/edit`, { method: 'POST', body: form, signal: AbortSignal.timeout(SPAWN_TIMEOUT_MS) });
  if (!spawn.ok) throw await detail(spawn, 'POST /edit');
  if (!(await spawn.json()).job_id) throw new Error('POST /edit returned no job_id');

  const started = Date.now();
  const status = await awaitJob(jobId, { timeoutMs: editTimeoutMs(), log });
  const files = await download(jobId, status.files);
  log(`${Object.keys(files).length} artifact(s), ${((Date.now() - started) / 60_000).toFixed(1)} min`);
  return { files, record: { job_id: jobId, ...(status.record ?? {}) } };
}
