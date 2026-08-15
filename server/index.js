import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import { Readable } from 'node:stream';
// Imported first because it is what reads .env, which is where PARTFIELD_BASE_URL and the
// directory overrides live rather than in whichever shell started the server.
import {
  MODELS_DIR,
  DATASET_DIR,
  GENERATED_DIR,
  PLACEMENT_RESULTS_DIR,
  EDIT_RESULTS_DIR,
  SEGMENT_RESULTS_DIR,
  P3SAM_RESULTS_DIR,
  FRONT3D_DIR,
  SCENE_EDITS_DIR,
  WEB_DIST_DIR,
} from '../pipeline/paths.mjs';
import { segment, cut } from '../pipeline/partfield.mjs';
import { segment as segmentP3SAM } from '../pipeline/p3sam.mjs';
import { readResult, writeCut, writeRecord } from '../pipeline/segments.mjs';
import { writeAtomic } from '../pipeline/metadata.mjs';
import {
  listScenes,
  sceneSummary,
  identifyScene,
  listRuns,
  readRun,
  readResults,
  createRun,
  setPrompt,
  setLabels,
  askModel,
  buildPrompt,
  models as editModels,
} from '../pipeline/scene-edit.mjs';

const PORT = Number(process.env.PORT ?? 3000);
const OPENROUTER_MODELS_URL = 'https://openrouter.ai/api/v1/models';
const OPENROUTER_MODELS_TTL_MS = Number(process.env.OPENROUTER_MODELS_TTL_MS ?? 5 * 60_000);
const OPENROUTER_MODELS_LIMIT = 80;
let openRouterModelsCache = { expires: 0, models: [], pending: null };

fs.mkdirSync(MODELS_DIR, { recursive: true });
fs.mkdirSync(DATASET_DIR, { recursive: true });
fs.mkdirSync(GENERATED_DIR, { recursive: true });
fs.mkdirSync(PLACEMENT_RESULTS_DIR, { recursive: true });
fs.mkdirSync(EDIT_RESULTS_DIR, { recursive: true });
fs.mkdirSync(SEGMENT_RESULTS_DIR, { recursive: true });
fs.mkdirSync(P3SAM_RESULTS_DIR, { recursive: true });
fs.mkdirSync(FRONT3D_DIR, { recursive: true });
fs.mkdirSync(SCENE_EDITS_DIR, { recursive: true });

const app = express();
app.use(express.json({ limit: '512mb' }));

function listModels(dir, base = '') {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const rel = base ? `${base}/${entry.name}` : entry.name;
    if (entry.isDirectory()) out.push(...listModels(path.join(dir, entry.name), rel));
    else if (/\.(glb|gltf)$/i.test(entry.name)) out.push(rel);
  }
  return out;
}

app.get('/api/models', (_req, res) => {
  try {
    res.json({ models: listModels(MODELS_DIR).sort((a, b) => a.localeCompare(b)) });
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
});

app.use('/models', express.static(MODELS_DIR));
app.use('/dataset', express.static(DATASET_DIR));
app.use('/generated', express.static(GENERATED_DIR));
app.use('/placement-results', express.static(PLACEMENT_RESULTS_DIR));
app.use('/edit-results', express.static(EDIT_RESULTS_DIR));
app.use('/segment-results', express.static(SEGMENT_RESULTS_DIR));
app.use('/p3sam-results', express.static(P3SAM_RESULTS_DIR));
app.use('/scenes', express.static(FRONT3D_DIR));
app.use('/scene-edits', express.static(SCENE_EDITS_DIR));

// A posed mesh is read out of the sample's own folder when it is there, and pulled back from
// the scene volume when it is not — baking writes to the volume, so whether a sample's GLBs
// are local depends on whether anyone brought them down. Going through the server either way
// keeps the viewer same-origin and spares it from knowing which of the two it got.
const SCENE_BASE_URL = process.env.SCENE_BASE_URL ?? 'https://starshot-aitools--dc-scene-ops-web.modal.run';

app.get('/mesh/:id/:name', async (req, res) => {
  const id = path.basename(req.params.id);
  const name = path.basename(req.params.name);

  const local = path.join(GENERATED_DIR, id, name);
  if (fs.existsSync(local)) return res.type('model/gltf-binary').sendFile(local);

  const target = `${SCENE_BASE_URL}/file/${encodeURIComponent(id)}/${encodeURIComponent(name)}`;
  try {
    const upstream = await fetch(target, { signal: AbortSignal.timeout(120_000) });
    if (!upstream.ok) return res.status(upstream.status).json({ error: `scene-ops ${upstream.status}` });
    res.type('model/gltf-binary');
    Readable.fromWeb(upstream.body).pipe(res);
  } catch (err) {
    res.status(502).json({ error: String(err) });
  }
});

const ROLES = ['anchor', 'placed'];

const readMetadata = (dir) => {
  const file = path.join(dir, 'metadata.json');
  return fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : null;
};

/**
 * The two GLBs a run loads, anchor first. A placed sample names them itself; a folder still on
 * the old `spec.json` format names nothing, so they are taken off disk and ordered by the one
 * convention both formats share — the anchor is the file named after the folder.
 */
function runMeshes(dir, id, metadata) {
  if (metadata) return ROLES.map((role) => metadata[role]?.mesh).filter(Boolean);
  const anchor = `${id}.glb`;
  return fs
    .readdirSync(dir)
    .filter((file) => /\.glb$/i.test(file))
    .sort((a, b) => (a === anchor ? -1 : b === anchor ? 1 : a.localeCompare(b)));
}

// Everything in data/generated/ with a pair of meshes to look at. A sample carries `combined_size`
// once pipeline/run.mjs has posed it, and describes itself from there. A spec.json folder is
// read for nothing but its id and shows its GLBs as they are, which for those is unposed.
app.get('/api/runs', (_req, res) => {
  try {
    const runs = [];
    for (const entry of fs.readdirSync(GENERATED_DIR, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const dir = path.join(GENERATED_DIR, entry.name);
      const metadata = readMetadata(dir);
      if (metadata && !Array.isArray(metadata.combined_size)) continue;

      const meshes = runMeshes(dir, entry.name, metadata);
      if (meshes.length !== 2) continue;
      runs.push({ id: entry.name, meshes, placed: !!metadata });
    }
    res.json({ runs: runs.sort((a, b) => a.id.localeCompare(b.id)) });
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
});

/**
 * The placement benchmark: every generated sample and, per sample, the models that
 * have posed it. A sample is listed from `generation.json` alone, so one whose meshes
 * exist but which no model has placed yet still shows up with an empty model list.
 */
app.get('/api/placements', (_req, res) => {
  try {
    const samples = [];
    for (const entry of fs.readdirSync(PLACEMENT_RESULTS_DIR, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const dir = path.join(PLACEMENT_RESULTS_DIR, entry.name);
      const generation = path.join(dir, 'generation.json');
      if (!fs.existsSync(generation)) continue;

      const models = [];
      for (const sub of fs.readdirSync(dir, { withFileTypes: true })) {
        if (!sub.isDirectory()) continue;
        const meta = path.join(dir, sub.name, 'placement.json');
        if (!fs.existsSync(meta)) continue;
        models.push({ slug: sub.name, model: JSON.parse(fs.readFileSync(meta, 'utf8')).model ?? sub.name });
      }

      const { placement = '', anchor = {}, placed = {} } = JSON.parse(fs.readFileSync(generation, 'utf8'));
      samples.push({
        id: entry.name,
        placement,
        anchor_view: anchor.view_image ?? null,
        generated_image: placed.generated_image ?? null,
        models: models.sort((a, b) => a.model.localeCompare(b.model)),
      });
    }
    res.json({ samples: samples.sort((a, b) => a.id.localeCompare(b.id)) });
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
});

/**
 * The editing experiment: every sample pipeline/test-edit.mjs has masked, and whether the GPU
 * half finished it too. Each phase is listed off the file it writes last — mask.json locally,
 * edit.json for the service — so a folder mid-write is never offered as ready. The 2D views
 * are read off disk rather than assumed, since a job hands back only the ones it produced.
 */
app.get('/api/edits', (_req, res) => {
  try {
    const samples = [];
    for (const entry of fs.readdirSync(EDIT_RESULTS_DIR, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const dir = path.join(EDIT_RESULTS_DIR, entry.name);
      const mask = path.join(dir, 'mask.json');
      if (!fs.existsSync(mask)) continue;

      const images = path.join(dir, 'images');
      samples.push({
        id: entry.name,
        placement: JSON.parse(fs.readFileSync(mask, 'utf8')).placement ?? '',
        edited: fs.existsSync(path.join(dir, 'edit.json')),
        views: fs.existsSync(images) ? fs.readdirSync(images).filter((file) => /\.png$/i.test(file)) : [],
      });
    }
    res.json({ samples: samples.sort((a, b) => a.id.localeCompare(b.id)) });
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
});

/**
 * PartField part segmentation, from the browser.
 *
 * The service is a couple of minutes of A10G for a mesh it has not seen, so a request starts a
 * job and is polled — the same shape the service itself presents, and the same shape the rest
 * of this file's Modal work takes. The job is only a progress bar: what is worth keeping is
 * written into data/segment-results/ as it lands, in the layout pipeline/test-segment.mjs writes, so
 * a mesh dropped into the page and a mesh named on the command line are afterwards the same
 * thing. Keyed by result id, since two segmentations of one mesh would be writing to one folder.
 */
const segmenting = new Map();

function startJob(jobs, id, work) {
  const job = { id, status: 'running', stage: 'queued', error: null, started: Date.now() };
  jobs.set(id, job);
  work((stage) => {
    job.stage = stage;
  })
    .then(() => Object.assign(job, { status: 'done', stage: 'done' }))
    .catch((err) => Object.assign(job, { status: 'failed', error: String(err?.message ?? err) }));
  return job;
}

app.get('/api/segments', (_req, res) => {
  try {
    const samples = [];
    for (const entry of fs.readdirSync(SEGMENT_RESULTS_DIR, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const result = readResult(path.join(SEGMENT_RESULTS_DIR, entry.name));
      if (result) samples.push({ id: result.id ?? entry.name, faces: result.faces, cuts: result.on_disk });
    }
    res.json({ samples: samples.sort((a, b) => a.id.localeCompare(b.id)) });
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
});

app.get('/api/segments/:id', (req, res) => {
  const id = sanitizeDirName(req.params.id);
  try {
    const result = readResult(path.join(SEGMENT_RESULTS_DIR, id));
    const job = segmenting.get(id) ?? null;
    if (!result && !job) return res.status(404).json({ error: `no segmentation for ${id}` });
    res.json({ result, job });
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
});

// The mesh arrives as the raw body rather than as base64 in JSON like /api/export: it is one
// file of tens of megabytes and there is no metadata to carry beside it, so the query string
// does for the rest.
app.post('/api/segments', express.raw({ type: () => true, limit: '512mb' }), (req, res) => {
  const name = path.basename(String(req.query.name ?? 'mesh.glb'));
  const id = sanitizeDirName(name.replace(/\.[^.]+$/, ''));
  if (!id) return res.status(400).json({ error: 'the mesh needs a usable name' });
  if (!req.body?.length) return res.status(400).json({ error: 'no mesh in the request body' });
  if (segmenting.get(id)?.status === 'running') return res.status(409).json({ error: `${id} is already segmenting` });

  const parts = Number(req.query.parts ?? 8);
  const maxClusters = Number(req.query.max_clusters ?? 20);
  const model = req.body;
  const dir = path.join(SEGMENT_RESULTS_DIR, id);

  startJob(segmenting, id, async (stage) => {
    const result = await segment({ id, model, filename: name, parts, maxClusters, log: stage });
    writeRecord(dir, { id, source: name, job: result.jobId, summary: result.summary, cuts: [await writeCut(dir, result)] });
  });
  res.json({ id });
});

// Another level of a hierarchy already built, which is seconds rather than minutes and never
// touches a GPU. What makes the viewer's parts slider worth dragging.
app.post('/api/segments/:id/cut', (req, res) => {
  const id = sanitizeDirName(req.params.id);
  const dir = path.join(SEGMENT_RESULTS_DIR, id);
  const record = readResult(dir);
  if (!record?.job) return res.status(404).json({ error: `no segmented job for ${id}` });
  if (segmenting.get(id)?.status === 'running') return res.status(409).json({ error: `${id} is already segmenting` });

  const parts = Number(req.body?.parts);
  if (!record.levels?.includes(parts)) {
    return res.status(400).json({ error: `${id} was clustered to ${record.levels?.at(-1)} parts, not ${parts}` });
  }

  startJob(segmenting, id, async (stage) => {
    const result = await cut({ jobId: record.job, parts, log: stage });
    writeRecord(dir, { ...record, summary: result.summary, cuts: [await writeCut(dir, result)] });
  });
  res.json({ id, parts });
});

/**
 * P3-SAM automatic segmentation. The upstream model cleans the mesh before labelling it, so
 * these results intentionally live apart from PartField's source-face-aligned hierarchies.
 */
const p3samJobs = new Map();

function readP3SAM(id) {
  const file = path.join(P3SAM_RESULTS_DIR, id, 'result.json');
  return fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : null;
}

app.get('/api/p3sam', (_req, res) => {
  try {
    const samples = fs
      .readdirSync(P3SAM_RESULTS_DIR, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => readP3SAM(entry.name))
      .filter(Boolean)
      .map(({ id, source, faces, num_parts, created_at }) => ({ id, source, faces, num_parts, created_at }))
      .sort((a, b) => b.created_at - a.created_at);
    res.json({ samples });
  } catch (err) {
    res.status(500).json({ error: String(err?.message ?? err) });
  }
});

app.get('/api/p3sam/:id', (req, res) => {
  const id = sanitizeDirName(req.params.id);
  try {
    const result = readP3SAM(id);
    const job = p3samJobs.get(id) ?? null;
    if (!result && !job) return res.status(404).json({ error: `no P3-SAM result for ${id}` });
    res.json({ result, job });
  } catch (err) {
    res.status(500).json({ error: String(err?.message ?? err) });
  }
});

app.post('/api/p3sam', express.raw({ type: () => true, limit: '512mb' }), (req, res) => {
  const name = path.basename(String(req.query.name ?? 'mesh.glb'));
  const stem = sanitizeDirName(name.replace(/\.[^.]+$/, ''));
  if (!stem) return res.status(400).json({ error: 'the mesh needs a usable name' });
  if (!req.body?.length) return res.status(400).json({ error: 'no mesh in the request body' });

  const postProcess = String(req.query.post_process ?? 'true') !== 'false';
  const threshold = Number(req.query.threshold ?? 0.95);
  const seed = Number(req.query.seed ?? 42);
  if (!(threshold > 0 && threshold <= 1)) return res.status(400).json({ error: 'threshold must be greater than 0 and at most 1' });
  if (!Number.isInteger(seed)) return res.status(400).json({ error: 'seed must be an integer' });

  const key = `${stem}-${postProcess ? 'post' : 'raw'}-${String(threshold).replace('.', '_')}-s${seed}`;
  const id = sanitizeDirName(key);
  if (p3samJobs.get(id)?.status === 'running') return res.status(409).json({ error: `${id} is already segmenting` });

  const model = req.body;
  const dir = path.join(P3SAM_RESULTS_DIR, id);
  startJob(p3samJobs, id, async (stage) => {
    const result = await segmentP3SAM({ id, model, filename: name, postProcess, threshold, seed, log: stage });
    fs.mkdirSync(dir, { recursive: true });
    writeAtomic(path.join(dir, 'parts.glb'), result.glb);
    writeAtomic(path.join(dir, 'labels.bin'), Buffer.from(result.labels.buffer));
    const record = { ...result.summary, id, source: name, job: result.jobId, created_at: Date.now() };
    writeAtomic(path.join(dir, 'result.json'), `${JSON.stringify(record, null, 2)}\n`);
  });
  res.json({ id });
});

/**
 * The scene-editing benchmark: a run is one scene and one prompt, and every model under test
 * answers that same prompt — see pipeline/scene-edit.mjs.
 *
 * A model call is a job rather than a request. A reasoning model spends tens of seconds on a
 * prompt like this and the page adds several tiles at once, so a request starts the call and
 * the page polls. What lands is written into data/scene-edits/ as it arrives, which is why a
 * tile closed and opened again costs nothing.
 */
const asking = new Map();
const identifyingScenes = new Map();

const jobsFor = (runId) =>
  [...asking.values()]
    .filter((job) => job.run === runId)
    .map(({ model, status, error }) => ({ model, status, error }));

// Measured per file rather than in one go, so a room this cannot read names itself instead of
// taking the whole list of scenes down with it.
app.get('/api/scenes', (_req, res) => {
  const scenes = listScenes().map((file) => {
    try {
      const scene = sceneSummary(file);
      const job = identifyingScenes.get(file);
      return {
        ...scene,
        job: job ? { status: job.status, stage: job.stage, error: job.error } : null,
      };
    } catch (err) {
      return { file, identified: false, objects: [], job: null, error: String(err?.message ?? err) };
    }
  });
  res.json({ scenes });
});

app.post('/api/scenes/:file/identify', (req, res) => {
  const file = path.basename(req.params.file);
  if (!listScenes().includes(file)) return res.status(404).json({ error: `no scene named ${file}` });
  if (identifyingScenes.get(file)?.status === 'running') {
    return res.status(409).json({ error: `${file} is already being identified` });
  }
  const job = startJob(identifyingScenes, file, (stage) => identifyScene(file, { force: !!req.body?.force, log: stage }));
  res.json({ file, job: { status: job.status } });
});

app.get('/api/scene-runs', (_req, res) => {
  try {
    const runs = listRuns().map((run) => ({
      id: run.id,
      prompt: run.prompt,
      scene: run.scene.file,
      objects: run.objects.length,
      models: readResults(run.id).map(({ model, prompt_hash }) => ({ model, stale: prompt_hash !== run.prompt_hash })),
    }));
    res.json({ runs, default_models: editModels() });
  } catch (err) {
    res.status(500).json({ error: String(err?.message ?? err) });
  }
});

function numberOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function modelSummary(model) {
  const id = String(model?.id ?? '');
  const provider = id.includes('/') ? id.slice(0, id.indexOf('/')) : '';
  const parameters = Array.isArray(model?.supported_parameters) ? model.supported_parameters : [];
  return {
    id,
    name: String(model?.name ?? id),
    provider,
    description: String(model?.description ?? ''),
    context_length: numberOrNull(model?.context_length),
    pricing: {
      prompt: numberOrNull(model?.pricing?.prompt),
      completion: numberOrNull(model?.pricing?.completion),
    },
    structured_outputs: parameters.includes('structured_outputs'),
    response_format: parameters.includes('response_format'),
    created: numberOrNull(model?.created),
    expiration_date: model?.expiration_date ?? null,
  };
}

async function openRouterModels() {
  if (Date.now() < openRouterModelsCache.expires && openRouterModelsCache.models.length) {
    return openRouterModelsCache.models;
  }
  if (openRouterModelsCache.pending) return openRouterModelsCache.pending;

  openRouterModelsCache.pending = (async () => {
    const response = await fetch(`${OPENROUTER_MODELS_URL}?output_modalities=text&sort=most-popular`, {
      headers: { accept: 'application/json' },
      signal: AbortSignal.timeout(30_000),
    });
    if (!response.ok) throw new Error(`OpenRouter models ${response.status}: ${(await response.text()).slice(0, 200)}`);
    const payload = await response.json();
    const models = (Array.isArray(payload?.data) ? payload.data : [])
      .map(modelSummary)
      .filter((model) => model.id);
    openRouterModelsCache = { expires: Date.now() + OPENROUTER_MODELS_TTL_MS, models, pending: null };
    return models;
  })();

  try {
    return await openRouterModelsCache.pending;
  } catch (err) {
    openRouterModelsCache.pending = null;
    if (openRouterModelsCache.models.length) return openRouterModelsCache.models;
    throw err;
  }
}

app.get('/api/scene-models', async (req, res) => {
  try {
    const query = String(req.query.q ?? '').trim().toLowerCase();
    const models = await openRouterModels();
    const filtered = query
      ? models.filter((model) =>
          `${model.name} ${model.id} ${model.provider}`.toLowerCase().includes(query),
        )
      : models;
    const available = filtered.filter((model) => !model.expiration_date);
    res.json({ models: available.slice(0, OPENROUTER_MODELS_LIMIT), total: available.length });
  } catch (err) {
    res.status(502).json({ error: String(err?.message ?? err) });
  }
});

// A run copies an identified scene's names and facing axes. Its instruction can be written
// afterwards, and names can still be corrected manually without changing the scene-wide cache.
app.post('/api/scene-runs', (req, res) => {
  try {
    res.json({ run: createRun(req.body ?? {}) });
  } catch (err) {
    res.status(400).json({ error: String(err?.message ?? err) });
  }
});

app.post('/api/scene-runs/:id/prompt', (req, res) => {
  try {
    res.json({ run: setPrompt(req.params.id, req.body?.prompt) });
  } catch (err) {
    res.status(400).json({ error: String(err?.message ?? err) });
  }
});

app.post('/api/scene-runs/:id/labels', (req, res) => {
  try {
    res.json({ run: setLabels(req.params.id, req.body?.labels) });
  } catch (err) {
    res.status(400).json({ error: String(err?.message ?? err) });
  }
});

// A whole run in one reply: the manifest the prompt is built from, the exact text every model
// was sent, every answer on disk and every call still in flight. One endpoint because the page
// polls it while a tile waits, and a tile's state is a function of all four.
app.get('/api/scene-runs/:id', (req, res) => {
  try {
    const run = readRun(req.params.id);
    const results = readResults(run.id).map((result) => ({ ...result, stale: result.prompt_hash !== run.prompt_hash }));
    res.json({ run, prompt: buildPrompt(run), results, jobs: jobsFor(run.id) });
  } catch (err) {
    res.status(404).json({ error: String(err?.message ?? err) });
  }
});

// Answers off disk unless `force` is set, so the page can post for every tile it opens and let
// the reply decide whether that costs a call.
app.post('/api/scene-runs/:id/models', (req, res) => {
  let run;
  try {
    run = readRun(req.params.id);
  } catch (err) {
    return res.status(404).json({ error: String(err?.message ?? err) });
  }

  const model = String(req.body?.model ?? '').trim();
  if (!model) return res.status(400).json({ error: 'no model in the request' });
  if (!run.prompt) return res.status(400).json({ error: `${run.id} has no instruction yet` });

  const key = `${run.id}::${model}`;
  if (asking.get(key)?.status === 'running') return res.status(409).json({ error: `${model} is already answering` });

  const job = startJob(asking, key, () => askModel(run, model, { force: !!req.body?.force }));
  Object.assign(job, { run: run.id, model });
  res.json({ run: run.id, model });
});

// Lists sample folders and the .glb files inside them. Reads directory entries
// only — it never opens or parses metadata.json.
app.get('/api/samples', (_req, res) => {
  try {
    const samples = fs
      .readdirSync(DATASET_DIR, { withFileTypes: true })
      .filter((e) => e.isDirectory())
      .map((e) => ({
        id: e.name,
        glbs: fs
          .readdirSync(path.join(DATASET_DIR, e.name))
          .filter((f) => /\.glb$/i.test(f))
          .sort((a, b) => a.localeCompare(b)),
      }))
      .filter((s) => s.glbs.length > 0)
      .sort((a, b) => a.id.localeCompare(b.id));
    res.json({ samples });
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
});

function nextSampleId() {
  const existing = fs
    .readdirSync(DATASET_DIR, { withFileTypes: true })
    .filter((e) => e.isDirectory() && /^sample_\d+$/.test(e.name))
    .map((e) => Number(e.name.slice('sample_'.length)));
  const next = existing.length ? Math.max(...existing) + 1 : 1;
  return `sample_${String(next).padStart(4, '0')}`;
}

function sanitizeDirName(name) {
  const base = path.basename(String(name ?? '')).trim().replace(/[^a-zA-Z0-9._-]+/g, '_');
  return base && base !== '.' && base !== '..' ? base : '';
}

app.post('/api/export', (req, res) => {
  try {
    const { dirName = '', placement = '', metadata = {}, files = {} } = req.body ?? {};
    const id = sanitizeDirName(dirName) || nextSampleId();
    const dir = path.join(DATASET_DIR, id);
    fs.mkdirSync(dir, { recursive: true });

    for (const [name, b64] of Object.entries(files)) {
      const safe = path.basename(String(name));
      fs.writeFileSync(path.join(dir, safe), Buffer.from(String(b64), 'base64'));
    }

    fs.writeFileSync(path.join(dir, 'placement.txt'), String(placement), 'utf8');
    fs.writeFileSync(
      path.join(dir, 'metadata.json'),
      JSON.stringify({ id, ...metadata }, null, 2),
      'utf8',
    );

    res.json({ id, path: dir });
  } catch (err) {
    res.status(500).json({ error: String(err?.stack ?? err) });
  }
});

if (fs.existsSync(path.join(WEB_DIST_DIR, 'index.html'))) {
  app.use(express.static(WEB_DIST_DIR));
}

app.listen(PORT, () => {
  console.log(`data-creator server → http://localhost:${PORT}`);
  console.log(`  models  : ${MODELS_DIR}`);
  console.log(`  dataset : ${DATASET_DIR}`);
  console.log(`  samples : ${GENERATED_DIR}`);
});
