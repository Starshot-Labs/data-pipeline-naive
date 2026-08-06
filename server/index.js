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

// A corpus is any folder at the repo root whose name starts with "generated" — each one a
// self-contained set of sample folders the viewer can browse. The env-chosen GENERATED_DIR
// stays the default whenever a request names no corpus (or names one that does not exist),
// which is also what keeps volume-backed setups working unchanged.
const CORPUS_NAME = /^generated[A-Za-z0-9._-]*$/;

function corpusDir(name) {
  if (typeof name === 'string' && CORPUS_NAME.test(name)) {
    const dir = path.join(ROOT, name);
    if (fs.existsSync(dir) && fs.statSync(dir).isDirectory()) return dir;
  }
  return GENERATED_DIR;
}

app.get('/api/corpora', (_req, res) => {
  try {
    const corpora = fs
      .readdirSync(ROOT, { withFileTypes: true })
      .filter((entry) => entry.isDirectory() && CORPUS_NAME.test(entry.name))
      .map((entry) => ({
        id: entry.name,
        samples: fs
          .readdirSync(path.join(ROOT, entry.name), { withFileTypes: true })
          .filter((child) => child.isDirectory()).length,
      }))
      .sort((a, b) => a.id.localeCompare(b.id));
    // '' when GENERATED_DIR lives outside the repo root — the picker shows it as "default".
    const fallback = path.dirname(GENERATED_DIR) === ROOT ? path.basename(GENERATED_DIR) : '';
    res.json({ corpora, default: corpora.some((c) => c.id === fallback) ? fallback : '' });
  } catch (err) {
    res.status(500).json({ error: String(err) });
  }
});

// Any file of one sample — metadata.json, reference images — out of the named corpus.
// Meshes have their own route below, which knows about posed copies and the scene volume.
app.get('/file/:corpus/:id/:name', (req, res) => {
  const dir = corpusDir(req.params.corpus);
  const file = path.join(dir, path.basename(req.params.id), path.basename(req.params.name));
  if (!fs.existsSync(file)) return res.status(404).json({ error: 'no such file' });
  res.sendFile(file, { dotfiles: 'allow' });
});

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

  const local = path.join(corpusDir(req.query.corpus), id, name);
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
  const corpus = corpusDir(req.body?.corpus);
  if (!id || !fs.existsSync(path.join(corpus, id))) {
    return res.status(404).json({ error: `no sample folder named "${id}"` });
  }
  const key = `${corpus}|${id}`;
  if (placing.has(key)) return res.status(409).json({ error: `already placing ${id}` });
  placing.add(key);

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
      `--source=${corpus}`,
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
    placing.delete(key);
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
  const dir = path.join(corpusDir(req.query.corpus), id);
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

// The full model conversation behind a sample's placements, read back from the audit logs
// pipeline/run.mjs writes: every log whose filename names this sample, parsed into its
// sections, newest first. The same id can be re-placed many times and can live in several
// corpora at once, so each log also answers whether its baked transforms are the ones the
// sample's posed mesh carries right now — matched, not guessed from recency.
const PLACEMENT_LOG_DIR = path.resolve(ROOT, process.env.PLACEMENT_LOG_DIR ?? 'placement-logs');

function parsePlacementLog(file) {
  const text = fs.readFileSync(file, 'utf8');
  const [head, ...rest] = text.split(/^## /m);
  const bullet = (name) => head.match(new RegExp(`^- ${name}: (.+)$`, 'm'))?.[1] ?? null;

  const sections = {};
  for (const chunk of rest) {
    const eol = chunk.indexOf('\n');
    sections[chunk.slice(0, eol).trim()] = chunk.slice(eol + 1).trim();
  }
  const fenced = (section = '') => section.match(/^```(?:json)?\n([\s\S]*?)\n```/m)?.[1] ?? section;

  let transforms = null;
  try {
    transforms = JSON.parse(fenced(sections['Resolved transforms (baked)'] ?? sections['Resolved transforms']));
  } catch {
    transforms = null; // an older or hand-edited log still shows its text sections
  }

  return {
    stamp: bullet('at'),
    model: bullet('model'),
    tokens: bullet('tokens'),
    cost: bullet('cost'),
    placement: (sections['Placement'] ?? '').replace(/^"([\s\S]*)"$/, '$1'),
    system: fenced(sections['System prompt'] ?? ''),
    user: fenced(sections['User prompt'] ?? ''),
    answer: sections['Answer'] ?? '',
    physics: sections['Physics'] ?? '',
    transforms,
  };
}

app.get('/api/placement-logs/:id', async (req, res) => {
  const id = path.basename(req.params.id);
  try {
    const files = fs.existsSync(PLACEMENT_LOG_DIR)
      ? fs.readdirSync(PLACEMENT_LOG_DIR).filter((file) => file.endsWith(`_${id}.md`)).sort().reverse()
      : [];
    const logs = files.map((file) => ({ file, ...parsePlacementLog(path.join(PLACEMENT_LOG_DIR, file)) }));

    // The pose on screen is whatever the posed placed mesh carries; the log that baked it
    // is the one whose resolved transforms agree. Volume-resident samples have no local
    // file to read, so their logs simply go unmarked.
    const dir = path.join(corpusDir(req.query.corpus), id);
    const placedMesh = readMetadata(dir)?.placed?.mesh;
    if (placedMesh) {
      const posed = path.join(dir, placedMesh.replace(/\.glb$/i, '.posed.glb'));
      const file = fs.existsSync(posed) ? posed : path.join(dir, placedMesh);
      if (fs.existsSync(file)) {
        const { parseGLB, extractTransform } = await import('../pipeline/glb.mjs');
        const baked = extractTransform(parseGLB(fs.readFileSync(file)));
        if (baked) {
          const close = (a = [], b = []) => a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) < 1e-4);
          for (const log of logs) {
            const placed = log.transforms?.placed;
            log.current = !!placed && close(placed.position, baked.position) &&
              close(placed.rotation, baked.rotation) && close(placed.scale, baked.scale);
          }
        }
      }
    }
    res.json({ logs });
  } catch (err) {
    res.status(500).json({ error: String(err?.message ?? err) });
  }
});

// Everything in generated/ with a pair of meshes to look at. A sample carries `combined_size`
// once pipeline/run.mjs has posed it, and describes itself from there. A spec.json folder is
// read for nothing but its id and shows its GLBs as they are, which for those is unposed.
app.get('/api/runs', (req, res) => {
  try {
    const corpus = corpusDir(req.query.corpus);
    const runs = [];
    for (const entry of fs.readdirSync(corpus, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const dir = path.join(corpus, entry.name);
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
