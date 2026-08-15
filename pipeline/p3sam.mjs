// HTTP client for the P3-SAM Modal service. A request starts an asynchronous GPU job;
// this module follows it to completion and downloads the segmented GLB, face labels and summary.

const DEFAULT_BASE_URL = 'https://starshot-aitools--dc-p3sam-web.modal.run';

const baseUrl = () => process.env.P3SAM_BASE_URL ?? DEFAULT_BASE_URL;
const pollMs = () => Number(process.env.P3SAM_POLL_S ?? 5) * 1000;
const timeoutMs = () => Number(process.env.P3SAM_TIMEOUT_S ?? 3600) * 1000;

const SPAWN_TIMEOUT_MS = 600_000;
const POLL_REQUEST_TIMEOUT_MS = 30_000;
const FILE_TIMEOUT_MS = 300_000;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const detail = async (response, label) =>
  new Error(`${label} ${response.status}: ${(await response.text()).slice(0, 500)}`);

export async function health() {
  const response = await fetch(`${baseUrl()}/health`, { signal: AbortSignal.timeout(30_000) });
  if (!response.ok) throw await detail(response, 'GET /health');
  return response.json();
}

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
    if (status.status === 'failed') throw new Error(`job ${jobId} failed: ${String(status.error).slice(0, 1000)}`);
  }
}

async function download(jobId, name) {
  const deadline = Date.now() + FILE_TIMEOUT_MS;
  for (;;) {
    const response = await fetch(`${baseUrl()}/jobs/${jobId}/file/${encodeURIComponent(name)}`, {
      signal: AbortSignal.timeout(POLL_REQUEST_TIMEOUT_MS),
    });
    if (response.ok) return Buffer.from(await response.arrayBuffer());
    if (response.status !== 404 || Date.now() >= deadline) throw await detail(response, `GET file ${name}`);
    await sleep(1000);
  }
}

export async function segment({ id = '', model, filename = 'mesh.glb', postProcess = true, threshold = 0.95, seed = 42, log = () => {} }) {
  const form = new FormData();
  form.append('model', new Blob([model], { type: 'application/octet-stream' }), filename);
  form.append('sample', id);
  form.append('post_process', String(postProcess));
  form.append('threshold', String(threshold));
  form.append('seed', String(seed));

  const spawn = await fetch(`${baseUrl()}/segment`, {
    method: 'POST',
    body: form,
    signal: AbortSignal.timeout(SPAWN_TIMEOUT_MS),
  });
  if (!spawn.ok) throw await detail(spawn, 'POST /segment');
  const jobId = (await spawn.json()).job_id;
  if (!jobId) throw new Error('POST /segment returned no job_id');
  log(`P3-SAM job ${jobId}`);

  const status = await awaitJob(jobId, { log });
  const [glb, labelBytes, summaryBytes] = await Promise.all([
    download(jobId, 'parts.glb'),
    download(jobId, 'labels.bin'),
    download(jobId, 'result.json'),
  ]);
  const labels = new Int32Array(labelBytes.buffer.slice(labelBytes.byteOffset, labelBytes.byteOffset + labelBytes.byteLength));
  const summary = JSON.parse(summaryBytes.toString('utf8'));
  log(`${summary.num_parts} parts over ${summary.faces} faces in ${status.record?.total_seconds}s`);
  return { jobId, glb, labels, summary };
}
