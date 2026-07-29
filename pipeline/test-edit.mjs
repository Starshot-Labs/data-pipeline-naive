// Experiment 2 · placement as 3D editing, following VoxHammer
// (github.com/Nelipot-Lee/VoxHammer) step for step.
//
// Experiment 1 generated the placed object as its own mesh and posed it beside the anchor.
// This one never makes a second mesh: the anchor is inverted into TRELLIS's latent space and
// re-denoised with the region around the target released, so the object grows out of the
// anchor's own asset and comes back as one edited GLB.
//
// The paper wants four inputs — original mesh, mask mesh, image, prompt — and a placement
// sample has three of them. The mask is derived here (mask.mjs): an LLM reads the anchor's
// 64³ occupancy and the phrase, returns one broad box covering wherever the object might
// generate, and the anchor's own voxels are subtracted from it.
//
//   local, this file          on the service (modal/voxhammer.py)
//   1 · normalize the anchor  3 · render 5 RGB views + the 2D mask by depth comparison
//   2 · box → mask mesh       4 · inpaint the chosen view (FLUX.1-Fill-dev)
//                             5 · 150-view render, DINOv2 features, voxels, voxels_delete
//                             6 · invert and re-denoise, decode to GLB
//
//   node pipeline/test-edit.mjs                 every sample in placement-set/
//   node pipeline/test-edit.mjs sample-4 ...    specific samples
//   node pipeline/test-edit.mjs --mask-only     stop after the mask, call no service
//   --force  --set=DIR  --out=DIR  --concurrency=N
//
// Both files handed to the service are in the cube TRELLIS works in — longest axis 1,
// centred on the origin — because VoxHammer normalizes the model but not the mask, and
// tests the mask against a fixed 64³ lattice over [-0.5, 0.5]³. Pre-normalizing both and
// telling its renderer to normalize nothing is what keeps the two in the same frame.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseGLB, serializeGLB, sceneTriangles, bakeTransform, transformTriangles } from './glb.mjs';
import { toSlices } from './voxelize.mjs';
import { renderView } from './render.mjs';
import { GRID, unitCube, lattice, requestBox, buildMask } from './mask.mjs';
import { mapLimit, retry, widthOf } from './limit.mjs';
import { writeAtomic } from './metadata.mjs';
import { listSamples, readSample, nameOf } from './samples.mjs';
import { editSample, health } from './voxhammer.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ENV_FILE = path.join(ROOT, '.env');
if (fs.existsSync(ENV_FILE)) process.loadEnvFile(ENV_FILE);

const args = process.argv.slice(2);
const flag = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

const SET_DIR = path.resolve(ROOT, flag('set', 'placement-set'));
const OUT_DIR = path.resolve(ROOT, flag('out', 'edit-results'));
const FORCE = args.includes('--force');
// Narrow by default, unlike the other stages: a worker holds the anchor's whole triangle
// soup and a 192³ sampling grid at once, and half a dozen of those exhaust the heap.
const WIDTH = Number(flag('concurrency', widthOf('EDIT_CONCURRENCY', 2)));
const MODEL = process.env.OPENROUTER_MODEL ?? 'google/gemini-3.1-pro-preview';

const MASK_FILE = 'mask.json';
const EDIT_FILE = 'edit.json';

const round = (v) => Number(v.toFixed(6));
const rounded = (arr) => arr.map(round);
const writeJson = (file, value) => writeAtomic(file, `${JSON.stringify(value, null, 2)}\n`);
const readJson = (file) => JSON.parse(fs.readFileSync(file, 'utf8'));

/** Steps 1-2: the two meshes VoxHammer takes, both in its cube, plus a look at them. */
async function buildInputs(id) {
  const say = (message) => console.log(`  [${id}] ${message}`);
  const { dir, anchorFile, imageFile, phrase } = readSample(SET_DIR, id);

  const out = path.join(OUT_DIR, id);
  if (FORCE) fs.rmSync(out, { recursive: true, force: true });
  if (fs.existsSync(path.join(out, MASK_FILE))) {
    say('mask already built — use --force to redo');
    return { out, phrase, imageFile, mask: readJson(path.join(out, MASK_FILE)) };
  }
  fs.mkdirSync(out, { recursive: true });
  say(`"${phrase}"  anchor=${anchorFile}  object=${imageFile}`);

  // 1 · the anchor, moved into the cube and baked so nothing downstream has to re-derive it
  const source = parseGLB(fs.readFileSync(path.join(dir, anchorFile)));
  const raw = sceneTriangles(source);
  const trs = unitCube(raw);
  const anchor = transformTriangles(raw, trs);
  writeAtomic(path.join(out, 'model.glb'), serializeGLB(bakeTransform(source, trs)));

  const occupancy = lattice(anchor);
  const filled = occupancy.reduce((total, cell) => total + cell, 0);
  say(`1 · normalized  scale=${round(trs.scale[0])}  ${filled} of ${GRID ** 3} cells solid`);

  // 2 · the box the object may grow into, minus the anchor itself
  const box = await retry(() => requestBox({
    phrase,
    object: nameOf(imageFile),
    anchor: { name: nameOf(anchorFile), grid: { dims: [GRID, GRID, GRID] }, slices: toSlices({ dims: [GRID, GRID, GRID], data: occupancy }) },
    model: MODEL,
  }));
  const mask = buildMask(box, occupancy);
  writeAtomic(path.join(out, 'mask.glb'), mask.glb);
  say(`2 · box [${box.min}]..[${box.max}]  ${mask.voxels} cells, ${mask.triangles} triangles`);

  // The anchor is ghosted and the mask drawn solid and dark: a mask for "inside" sits
  // behind the anchor's near wall, so a solid anchor hides the very thing worth checking.
  // Re-reading the GLB also proves the file we just wrote parses.
  writeAtomic(path.join(out, 'mask-view.png'), await renderView([
    { triangles: anchor, dither: true },
    { triangles: sceneTriangles(parseGLB(mask.glb)), base: 90 },
  ]));
  writeAtomic(path.join(out, 'source.png'), fs.readFileSync(path.join(dir, imageFile)));
  writeAtomic(path.join(out, 'placement.txt'), phrase);

  const record = {
    id,
    placement: phrase,
    created_at: new Date().toISOString(),
    model: box.model,
    anchor: { source: anchorFile, mesh: 'model.glb', normalization: { position: rounded(trs.position), scale: round(trs.scale[0]) }, solid_cells: filled },
    object: { source_image: imageFile, name: nameOf(imageFile) },
    mask: {
      mesh: 'mask.glb',
      grid: GRID,
      box: { min: box.min, max: box.max },
      cells: mask.voxels,
      triangles: mask.triangles,
      volume: round(mask.volume),
      reasoning: box.reasoning,
    },
    preview: 'mask-view.png',
    usage: box.usage,
  };
  writeJson(path.join(out, MASK_FILE), record);
  return { out, phrase, imageFile, mask: record };
}

/** Steps 3-6, all of which need a GPU, so all of which happen on the service. */
async function runEdit(id, { out, phrase, mask }) {
  const say = (message) => console.log(`  [${id}] ${message}`);
  if (!FORCE && fs.existsSync(path.join(out, EDIT_FILE))) {
    say('already edited — use --force to redo');
    return;
  }

  const result = await editSample({
    id,
    model: fs.readFileSync(path.join(out, 'model.glb')),
    mask: fs.readFileSync(path.join(out, 'mask.glb')),
    prompt: phrase,
    log: (message) => say(`    ${message}`),
  });

  for (const [name, bytes] of Object.entries(result.files)) {
    const file = path.join(out, name);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    writeAtomic(file, bytes);
  }

  // The edit comes back as one asset, so there is nothing to compose — this is just a look
  // at it. Note it arrives in TRELLIS's export frame rather than the frame `model.glb` was
  // sent in, so it is not directly comparable to mask-view.png.
  const edited = sceneTriangles(parseGLB(result.files['output.glb']));
  writeAtomic(path.join(out, 'result-view.png'), await renderView([{ triangles: edited }]));

  writeJson(path.join(out, EDIT_FILE), {
    id,
    placement: phrase,
    mask_cells: mask.mask.cells,
    result: { mesh: 'output.glb', preview: 'result-view.png', triangles: edited.length / 9 },
    ...result.record,
  });
  say(`edited → ${path.relative(ROOT, out)}`);
}

const requested = args.filter((arg) => !arg.startsWith('--'));
const ids = requested.length ? requested : listSamples(SET_DIR);
console.log(`${ids.length} sample(s) from ${path.relative(ROOT, SET_DIR)} → ${path.relative(ROOT, OUT_DIR)}`);

// The masks are worth building whether or not the GPU half is up, so an unreachable service
// downgrades the run rather than failing it.
let maskOnly = args.includes('--mask-only');
if (!maskOnly) {
  try {
    console.log(`voxhammer → ${JSON.stringify(await health())}`);
  } catch (err) {
    console.log(`voxhammer unreachable (${err.message}) — building masks only`);
    maskOnly = true;
  }
}

async function runSample(id) {
  const inputs = await buildInputs(id);
  if (!maskOnly) await runEdit(id, inputs);
}

const results = await mapLimit(ids, WIDTH, async (id) => {
  try {
    await runSample(id);
    return true;
  } catch (err) {
    console.error(`  [${id}] ✗ ${err.message}`);
    return false;
  }
});

const done = results.filter(Boolean).length;
console.log(`\n${done}/${ids.length} ${maskOnly ? 'masked' : 'edited'}`);
if (done < ids.length) process.exitCode = 1;
