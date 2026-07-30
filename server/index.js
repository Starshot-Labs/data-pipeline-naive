import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { Readable } from 'node:stream';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

const MODELS_DIR = process.env.MODELS_DIR ? path.resolve(process.env.MODELS_DIR) : path.join(ROOT, 'models');
const DATASET_DIR = process.env.DATASET_DIR ? path.resolve(process.env.DATASET_DIR) : path.join(ROOT, 'dataset');
const GENERATED_DIR = path.resolve(ROOT, process.env.GENERATED_DIR ?? 'generated');
const DIST_DIR = path.join(ROOT, 'dist');
const PORT = Number(process.env.PORT ?? 3000);

fs.mkdirSync(MODELS_DIR, { recursive: true });
fs.mkdirSync(DATASET_DIR, { recursive: true });
fs.mkdirSync(GENERATED_DIR, { recursive: true });

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

// A posed mesh is read out of the sample's own folder when it is there, and pulled back from
// the scene volume when it is not — baking writes to the volume, so whether a sample's GLBs
// are local depends on whether anyone brought them down. Going through the server either way
// keeps the viewer same-origin and spares it from knowing which of the two it got.
const SCENE_BASE_URL = process.env.SCENE_BASE_URL ?? 'https://starshot-aitools--dc-scene-ops-web.modal.run';

app.get('/mesh/:id/:name', async (req, res) => {
  const id = path.basename(req.params.id);
  const name = path.basename(req.params.name);

  const local = path.join(GENERATED_DIR, id, name);
  // A locally re-placed sample keeps its raw mesh and gains a `.posed.glb` sibling; the
  // posed one is the sample's current answer, so it is the one served — unless the caller
  // asks for the original with `?raw=1`, which is how the viewer's Reset shows the pair
  // as it was before any placement.
  const posed = local.replace(/\.glb$/i, '.posed.glb');
  const file = !('raw' in req.query) && /\.glb$/i.test(name) && fs.existsSync(posed) ? posed : local;
  // dotfiles: a GENERATED_DIR living under a dot-folder is legitimate, and sendFile's
  // default of ignoring any dotted path segment would 404 every mesh inside one.
  if (fs.existsSync(file)) return res.type('model/gltf-binary').sendFile(file, { dotfiles: 'allow' });

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

// The models the placement picker offers: OpenRouter's public catalog, trimmed to models
// that can actually answer a structured-output placement call, cached in memory because the
// catalog is a couple hundred KB and changes rarely. The default mirrors place.mjs.
let modelCatalog = { at: 0, models: [] };

app.get('/api/placement-models', async (_req, res) => {
  try {
    if (Date.now() - modelCatalog.at > 60 * 60 * 1000) {
      const upstream = await fetch('https://openrouter.ai/api/v1/models', { signal: AbortSignal.timeout(30_000) });
      if (!upstream.ok) throw new Error(`openrouter models: HTTP ${upstream.status}`);
      const { data } = await upstream.json();
      const models = (data ?? [])
        .filter((model) => model.supported_parameters?.includes('structured_outputs'))
        .map((model) => ({ id: model.id, name: model.name, context_length: model.context_length }))
        .sort((a, b) => a.id.localeCompare(b.id));
      modelCatalog = { at: Date.now(), models };
    }
    res.json({
      models: modelCatalog.models,
      default: process.env.OPENROUTER_MODEL ?? 'google/gemini-3.1-pro-preview',
    });
  } catch (err) {
    res.status(502).json({ error: String(err?.message ?? err) });
  }
});

// One placement child per sample at a time — a second click while one runs is answered 409
// rather than racing it over the same files.
const placing = new Set();

// Placement from the viewer: runs the placement stages on one sample by spawning the CLI,
// which loads .env itself and owns all the failure handling — the server never needs an
// API key in its own environment. Three modes, one per button: `place` is the LLM alone
// (physics forced off), `physics` re-runs just the solver on the sample's baked pose, and
// `full` is both. `--force` because clicking a placing button *is* the request to re-place.
app.post('/api/place', (req, res) => {
  const id = path.basename(String(req.body?.id ?? ''));
  if (!id || !fs.existsSync(path.join(GENERATED_DIR, id))) {
    return res.status(404).json({ error: `no sample folder named "${id}"` });
  }
  if (placing.has(id)) return res.status(409).json({ error: `already placing ${id}` });
  placing.add(id);

  const mode = ['place', 'physics', 'full'].includes(req.body?.mode) ? req.body.mode : 'full';

  // The picker's choices ride in as env vars: the CLI's env-file load never overrides
  // variables that already exist, so these win over .env for just this child.
  const model = typeof req.body?.model === 'string' && /^[~\w.:/-]{1,128}$/.test(req.body.model)
    ? req.body.model
    : null;
  const resolution = clampResolution(req.body?.resolution);
  const reasoning = ['off', 'low', 'medium', 'high'].includes(req.body?.reasoning) ? req.body.reasoning : null;
  const contact = ['rest', 'lean', 'attach', 'embed', 'drape', 'none'].includes(req.body?.contact)
    ? req.body.contact
    : null;

  const env = { ...process.env };
  if (model) env.OPENROUTER_MODEL = model;
  if (resolution) {
    env.VOXEL_RES_ANCHOR = String(resolution);
    env.VOXEL_RES_PLACED = String(resolution);
  }
  if (reasoning) env.OPENROUTER_REASONING = reasoning;
  if (contact) env.PLACEMENT_CONTACT = contact;
  // Explicit both ways, so the buttons mean what they say regardless of what .env says.
  if (mode !== 'physics') env.PLACEMENT_PHYSICS = mode === 'full' ? 'on' : 'off';
  // Ask the CLI for its one structured line: the pose physics started from, which the
  // viewer overlays as a translucent ghost of the model's answer.
  env.PLACEMENT_GHOST = '1';

  const child = spawn(
    process.execPath,
    [
      path.join(ROOT, 'pipeline', 'run.mjs'),
      mode === 'physics' ? '--physics-only' : '--force',
      `--source=${GENERATED_DIR}`,
      id,
    ],
    { cwd: ROOT, env },
  );

  let log = '';
  child.stdout.on('data', (chunk) => { log += chunk; });
  child.stderr.on('data', (chunk) => { log += chunk; });

  let done = false;
  const finish = (status, body) => {
    if (done) return;
    done = true;
    placing.delete(id);
    res.status(status).json(body);
  };
  child.on('error', (err) => finish(500, { error: String(err) }));
  child.on('close', (code) => {
    // The child's structured line, parsed before the log is trimmed for display: the
    // placed object's transform from before the physics pass, for the viewer's ghost.
    let ghost;
    try {
      const line = [...log.matchAll(/^GHOST (\{.*\})\s*$/gm)].at(-1);
      if (line) ghost = JSON.parse(line[1]);
    } catch {
      ghost = undefined;
    }
    finish(200, { id, ok: code === 0, log: log.slice(-4000), ...(ghost ? { ghost } : {}) });
  });
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

/** A voxel resolution the pipeline can reasonably take, or null for "use the default". */
function clampResolution(value) {
  const n = Math.round(Number(value));
  return Number.isFinite(n) && n >= 4 && n <= 128 ? n : null;
}

// The voxel geometry the placement model reads: both meshes voxelized on demand at the
// pipeline's own resolutions (or the caller's `?res=` override), answered as the maximal
// blocks the prompt carries plus the raw occupancy they were merged from. Only a local
// corpus can answer this — a sample whose meshes live on the volume voxelizes there.
app.get('/api/blocks/:id', async (req, res) => {
  const id = path.basename(req.params.id);
  const dir = path.join(GENERATED_DIR, id);
  try {
    const meshes = fs.existsSync(dir) ? runMeshes(dir, id, readMetadata(dir)) : [];
    if (meshes.length !== 2 || !meshes.every((mesh) => fs.existsSync(path.join(dir, mesh)))) {
      return res.status(404).json({ error: 'this sample\'s raw meshes are not in its local folder' });
    }

    const [{ parseGLB, sceneTriangles }, { voxelize, toBlocks }] = await Promise.all([
      import('../pipeline/glb.mjs'),
      import('../pipeline/voxelize.mjs'),
    ]);

    const override = clampResolution(req.query.res);
    const grids = {};
    for (const [i, role] of ROLES.entries()) {
      const resolution = override ?? Number(process.env[i === 0 ? 'VOXEL_RES_ANCHOR' : 'VOXEL_RES_PLACED'] ?? 16);
      const grid = voxelize(sceneTriangles(parseGLB(fs.readFileSync(path.join(dir, meshes[i])))), resolution);

      const [dx, dy, dz] = grid.dims;
      const voxels = [];
      for (let y = 0; y < dy; y++)
        for (let z = 0; z < dz; z++)
          for (let x = 0; x < dx; x++) if (grid.data[(y * dz + z) * dx + x]) voxels.push([x, y, z]);

      grids[role] = {
        dims: grid.dims,
        voxelSize: grid.voxelSize,
        origin: grid.origin,
        center: grid.center,
        size: grid.size,
        blocks: toBlocks(grid),
        voxels,
      };
    }
    res.json(grids);
  } catch (err) {
    res.status(500).json({ error: String(err?.message ?? err) });
  }
});

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
