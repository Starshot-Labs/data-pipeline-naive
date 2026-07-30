// Client for `dc-voxhammer`, the GPU half of experiment 2.
//
// Same job shape as the single-object Trellis endpoint: POST /edit spawns, /jobs/{id} is
// polled, and the artifacts are pulled one file at a time. A run renders 150 views under
// Cycles, extracts DINOv2 features, inverts the asset and re-denoises it, so it is measured
// in tens of minutes — far too long to hold a response open.

const DEFAULT_BASE_URL = 'https://starshot-aitools--dc-voxhammer-web.modal.run';

// Read lazily: this module is evaluated before the entry point loads .env.
const baseUrl = () => process.env.VOXHAMMER_BASE_URL ?? DEFAULT_BASE_URL;
const pollMs = () => Number(process.env.VOXHAMMER_POLL_S ?? 15) * 1000;
const pollTimeoutMs = () => Number(process.env.VOXHAMMER_TIMEOUT_S ?? 7200) * 1000;

// The anchor meshes run 17-26 MB each, so this deadline covers an upload, not a handshake:
// several posted at once share the uplink and every one of them finishes late.
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

/**
 * One sample through the paper's GPU steps. Returns every artifact the run produced —
 * the chosen render, its 2D mask, the inpainted target and the edited GLB — plus what
 * the service recorded about the run.
 */
export async function editSample({ id, model, mask, prompt, log = () => {} }) {
  const form = new FormData();
  form.append('model', new Blob([model], { type: 'model/gltf-binary' }), 'model.glb');
  form.append('mask', new Blob([mask], { type: 'model/gltf-binary' }), 'mask.glb');
  form.append('prompt', prompt);
  form.append('sample', id);

  const started = Date.now();
  const spawn = await fetch(`${baseUrl()}/edit`, {
    method: 'POST',
    body: form,
    signal: AbortSignal.timeout(SPAWN_TIMEOUT_MS),
  });
  if (!spawn.ok) throw await detail(spawn, 'POST /edit');

  const jobId = (await spawn.json()).job_id;
  if (!jobId) throw new Error('POST /edit returned no job_id');
  log(`job ${jobId}`);

  const deadline = Date.now() + pollTimeoutMs();
  let stage = null;
  let status;
  for (let polls = 1; ; polls++) {
    if (Date.now() >= deadline) throw new Error(`job ${jobId} timed out after ${pollTimeoutMs() / 1000}s`);
    await sleep(pollMs());

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

    // The service names the step it is on, which is the only useful thing to print during
    // a run this long.
    if (status.stage && status.stage !== stage) {
      stage = status.stage;
      log(`${stage} (${((Date.now() - started) / 60_000).toFixed(1)} min)`);
    }
    if (status.status === 'done') break;
    if (status.status === 'failed') throw new Error(`job ${jobId} failed: ${String(status.error).slice(0, 500)}`);
  }

  const files = {};
  for (const name of status.files ?? []) {
    const response = await fetch(`${baseUrl()}/jobs/${jobId}/file/${encodeURIComponent(name)}`, {
      signal: AbortSignal.timeout(FILE_TIMEOUT_MS),
    });
    if (!response.ok) throw await detail(response, `GET file ${name}`);
    files[name] = Buffer.from(await response.arrayBuffer());
  }
  log(`${Object.keys(files).length} artifact(s), ${((Date.now() - started) / 60_000).toFixed(1)} min`);

  return { files, record: { job_id: jobId, ...(status.record ?? {}) } };
}
