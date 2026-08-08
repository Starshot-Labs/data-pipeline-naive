import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import { Readable } from 'node:stream';
import { fileURLToPath } from 'node:url';
import { segment, cut } from '../pipeline/partfield.mjs';
import { readResult, writeCut, writeRecord } from '../pipeline/segments.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

// The pipeline's own entry points do this too — it is what lets PARTFIELD_BASE_URL and the
// directory overrides be set in one place rather than in whichever shell started the server.
const ENV_FILE = path.join(ROOT, '.env');
if (fs.existsSync(ENV_FILE)) process.loadEnvFile(ENV_FILE);

const MODELS_DIR = process.env.MODELS_DIR ? path.resolve(process.env.MODELS_DIR) : path.join(ROOT, 'models');
const DATASET_DIR = process.env.DATASET_DIR ? path.resolve(process.env.DATASET_DIR) : path.join(ROOT, 'dataset');
const GENERATED_DIR = path.resolve(ROOT, process.env.GENERATED_DIR ?? 'generated');
const PLACEMENT_RESULTS_DIR = path.resolve(ROOT, process.env.PLACEMENT_RESULTS_DIR ?? 'placement-results');
const EDIT_RESULTS_DIR = path.resolve(ROOT, process.env.EDIT_RESULTS_DIR ?? 'edit-results');
const SEGMENT_RESULTS_DIR = path.resolve(ROOT, process.env.SEGMENT_RESULTS_DIR ?? 'segment-results');
const DIST_DIR = path.join(ROOT, 'dist');
const PORT = Number(process.env.PORT ?? 3000);

fs.mkdirSync(MODELS_DIR, { recursive: true });
fs.mkdirSync(DATASET_DIR, { recursive: true });
fs.mkdirSync(GENERATED_DIR, { recursive: true });
fs.mkdirSync(PLACEMENT_RESULTS_DIR, { recursive: true });
fs.mkdirSync(EDIT_RESULTS_DIR, { recursive: true });
fs.mkdirSync(SEGMENT_RESULTS_DIR, { recursive: true });

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

// Everything in generated/ with a pair of meshes to look at. A sample carries `combined_size`
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
 * written into segment-results/ as it lands, in the layout pipeline/test-segment.mjs writes, so
 * a mesh dropped into the page and a mesh named on the command line are afterwards the same
 * thing. Keyed by result id, since two segmentations of one mesh would be writing to one folder.
 */
const segmenting = new Map();

function startJob(id, work) {
  const job = { id, status: 'running', stage: 'queued', error: null, started: Date.now() };
  segmenting.set(id, job);
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

  startJob(id, async (stage) => {
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

  startJob(id, async (stage) => {
    const result = await cut({ jobId: record.job, parts, log: stage });
    writeRecord(dir, { ...record, summary: result.summary, cuts: [await writeCut(dir, result)] });
  });
  res.json({ id, parts });
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

if (fs.existsSync(path.join(DIST_DIR, 'index.html'))) {
  app.use(express.static(DIST_DIR));
}

app.listen(PORT, () => {
  console.log(`data-creator server → http://localhost:${PORT}`);
  console.log(`  models  : ${MODELS_DIR}`);
  console.log(`  dataset : ${DATASET_DIR}`);
  console.log(`  samples : ${GENERATED_DIR}`);
});
