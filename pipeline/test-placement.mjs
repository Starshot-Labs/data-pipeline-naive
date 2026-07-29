// The single-anchor placement paradigm, benchmarked over placement-set/ across
// several LLMs. A sample gives only an anchor GLB, a photo of the object to place
// and the placement phrase; the placed mesh does not exist yet and is generated.
//
// Two halves, because they cost different things and only one of them is per-model:
//
//   generation · once per sample          placement · once per sample per model
//   1 · isometric shot of the anchor      4 · voxelize anchor + generated mesh
//   2 · nano banana re-poses the photo    5 · the LLM picks scale, rotation, centre
//   3 · Trellis 2 meshes that image       6 · bake it into the generated GLB
//                                         7 · write that model's result folder
//
//   node pipeline/test-placement.mjs      both halves
//   node pipeline/test-generate.mjs       generation only
//   node pipeline/test-place.mjs          placement only, over existing meshes
//
//   <sampleId> ...   only these samples
//   --models=a,b     override the models under test
//   --force          redo what the script owns: generation re-meshes (which
//                    invalidates every model result), placement re-asks the models
//   --set=DIR --out=DIR --generate-concurrency=N --place-concurrency=N
//
// Results, with everything a model does not decide shared between them:
//
//   placement-results/<sample>/
//     anchor.glb  anchor-view.png  placed-generated.png  placed-raw.glb
//     generation.json                          written last, marks the half done
//     <model-slug>/placed.glb  scene-view.png  placement.json
//
// Sharing the mesh across models is the point: a placement comparison only means
// something when every model reasoned over identical inputs. Artifacts persist as
// they land and are reused, so an interrupted run resumes instead of re-paying.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Euler, MathUtils, Quaternion, Vector3 } from 'three';
import { parseGLB, serializeGLB, sceneTriangles, bakeTransform, forceOpaqueMaterials, transformTriangles } from './glb.mjs';
import { voxelize, toSlices } from './voxelize.mjs';
import { renderView, ISO_VIEW } from './render.mjs';
import { generateImage } from './nano-banana.mjs';
import { generateMesh, health } from './trellis.mjs';
import { buildPrompt } from './place.mjs';
import { chatJSON } from './openrouter.mjs';
import { mapLimit, retry, widthOf } from './limit.mjs';
import { writeAtomic } from './metadata.mjs';
import { listSamples as listIn, nameOf, readSample } from './samples.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ENV_FILE = path.join(ROOT, '.env');
if (fs.existsSync(ENV_FILE)) process.loadEnvFile(ENV_FILE);

const args = process.argv.slice(2);
const flag = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

const SET_DIR = path.resolve(ROOT, flag('set', 'placement-set'));
const OUT_DIR = path.resolve(ROOT, flag('out', 'placement-results'));
const FORCE = args.includes('--force');

// The models under benchmark. Every one is asked the same question about the same
// meshes, so the only variable is the model.
const DEFAULT_MODELS = [
  'openai/gpt-5.6-sol-pro',
  'moonshotai/kimi-k3',
  'anthropic/claude-fable-5',
  'google/gemini-3.1-pro-preview',
];
const MODELS = (flag('models', process.env.PLACEMENT_MODELS) ?? DEFAULT_MODELS.join(','))
  .split(',')
  .map((model) => model.trim())
  .filter(Boolean);

// Wide by default: a sample's Trellis job and a model's placement call are both
// waits, and the whole point of splitting the halves is to spend those at once.
const GENERATE_WIDTH = Number(flag('generate-concurrency', widthOf('TEST_GENERATE_CONCURRENCY')));
const PLACE_WIDTH = Number(flag('place-concurrency', widthOf('TEST_PLACE_CONCURRENCY')));

const RESOLUTION = {
  anchor: Number(process.env.VOXEL_RES_ANCHOR ?? 64),
  placed: Number(process.env.VOXEL_RES_PLACED ?? 32),
};
const IMAGE_MODEL = process.env.IMAGE_MODEL ?? 'gemini-3.1-flash-image-preview';

const GENERATION_FILE = 'generation.json';
const PLACEMENT_FILE = 'placement.json';

const round = (v) => Number(v.toFixed(6));
const rounded = (arr) => arr.map(round);

const MIME = { '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp' };
const mimeOf = (file) => MIME[path.extname(file).toLowerCase()] ?? 'image/png';

/** `openai/gpt-5.6-sol-pro` → `openai-gpt-5.6-sol-pro`, the folder it writes to. */
export const modelSlug = (model) => model.toLowerCase().replace(/[^a-z0-9.+-]+/g, '-').replace(/^-+|-+$/g, '');

export const listSamples = () => listIn(SET_DIR);

// ── step 2 · the re-posing prompt ────────────────────────────────────────────

function generationPrompt(phrase) {
  return [
    'Image 1 is a photo of object1. Image 2 is a render of object2, given only as spatial context.',
    `Generate an image of object1, transformed to look as though it were "${phrase}"`,
    'relative to object2 — the pose, orientation, tilt and deformation it would have once placed,',
    'while staying the exact same object: same shape, material, texture and colours as in image 1.',
    'BUT the picture must contain ONLY object1, completely isolated against a plain flat white background:',
    'object2 is not present at all, no surface, no container, nothing at all from image 2. Do not merge the',
    'two objects or reshape object1 into object2. AGAIN object1 is placed AS IF object2 were there but in',
    `the generated image object1 should be alone, posed AS IF it were "${phrase}".`,
    'Seen from a similar three-quarter viewpoint to image 2. note. the object might not need to transformed at all, in that case, just return object1 back',
  ].join(' ');
}

// ── step 5 · placement with rotation about every axis ────────────────────────

const SYSTEM = `You are a 3D scene composer. You get voxel occupancy grids for an anchor object A and a movable object B, plus a phrase describing how B should be placed relative to A. Return the transform that places B.

Coordinates are right-handed with +Y up and +Z as an object's front face (glTF convention).

Reading a grid: it is printed as horizontal slices, top layer first. Inside a slice each line is one Z index (first line z=0, last line z=dz-1) and each character is one X index starting at x=0; '#' is solid, '.' is empty. Cells are cubes, so a grid's dimensions are proportional to the object's real proportions. Consecutive layers that are identical are printed once under a range header like \`y=31..12\`, meaning every layer from 31 down to 12 looks like that.

Working units are A-voxels: A's grid is the frame, so cell (i,j,k) of A spans [i,i+1] x [j,j+1] x [k,k+1] and A's grid occupies [0,adx] x [0,ady] x [0,adz]. Coordinates are continuous, may be fractional, and may fall outside A's grid.

B's grid only tells you B's shape and proportions — its absolute size is arbitrary and you decide it. \`scale\` is how many A-voxels one B-voxel becomes, so B's bounding box ends up bdx*scale by bdy*scale by bdz*scale A-voxels.

B's mesh was generated from an image of the object already posed for this placement, so its grid may already carry the right tilt or deformation — rotate only as much as the grid you actually see requires.

Answer with:
- reasoning: pick B's real-world size relative to A and convert it to \`scale\`, read the target surface or cavity off A's slices, then derive the centre. Do the arithmetic here.
- scale
- yaw_degrees, pitch_degrees, roll_degrees: B's rotation, right-hand rule about each axis, applied about B's bounding-box centre in fixed world axes — roll about +Z first, then pitch about +X, then yaw about +Y. Most placements need only yaw; reach for pitch and roll when the placement demands a lean, a tilt or a lie-down.
- center: B's bounding-box centre in A-voxel coordinates. B rotates about that centre, so rotation never moves it.

Contact must be physically plausible: something resting on a surface touches it with no gap and no interpenetration, something inside a cavity fits within its walls and rests on its floor, something mounted touches the face it attaches to.`;

const SCHEMA = {
  type: 'object',
  properties: {
    reasoning: { type: 'string' },
    scale: { type: 'number' },
    yaw_degrees: { type: 'number' },
    pitch_degrees: { type: 'number' },
    roll_degrees: { type: 'number' },
    center: { type: 'array', items: { type: 'number' } },
  },
  required: ['reasoning', 'scale', 'yaw_degrees', 'pitch_degrees', 'roll_degrees', 'center'],
  additionalProperties: false,
};

async function requestPlacement({ phrase, a, b, model }) {
  const { data, model: used, usage } = await chatJSON({
    model,
    system: SYSTEM,
    user: buildPrompt({ phrase, a, b }),
    name: 'placement',
    schema: SCHEMA,
  });

  const valid =
    Number.isFinite(data.scale) && data.scale > 0 &&
    [data.yaw_degrees, data.pitch_degrees, data.roll_degrees].every(Number.isFinite) &&
    Array.isArray(data.center) && data.center.length === 3 && data.center.every(Number.isFinite);
  if (!valid) throw new Error(`invalid placement from model: ${JSON.stringify(data).slice(0, 500)}`);

  return { ...data, used, usage };
}

// ── step 6 · resolve the answer into the anchor's own frame ──────────────────

/**
 * The anchor stays exactly as its file has it. A-voxel coordinates map into that
 * frame through the grid's placement: its min corner sits at the anchor's bbox
 * centre plus the bbox-centred `origin`. B is scaled, rotated about its own bbox
 * centre (Euler YXZ — fixed-axis roll, then pitch, then yaw, matching the
 * prompt), then moved so that centre lands on `center`.
 */
function resolveTransform(anchor, placed, answer) {
  const scale = (answer.scale * anchor.grid.voxelSize) / placed.grid.voxelSize;
  const rotation = new Quaternion().setFromEuler(new Euler(
    MathUtils.degToRad(answer.pitch_degrees),
    MathUtils.degToRad(answer.yaw_degrees),
    MathUtils.degToRad(answer.roll_degrees),
    'YXZ',
  ));
  const target = new Vector3()
    .fromArray(answer.center)
    .multiplyScalar(anchor.grid.voxelSize)
    .add(new Vector3().fromArray(anchor.grid.center))
    .add(new Vector3().fromArray(anchor.grid.origin));
  const centreOffset = new Vector3().fromArray(placed.grid.center).multiplyScalar(scale).applyQuaternion(rotation);

  return {
    position: rounded(target.sub(centreOffset).toArray()),
    rotation: rounded(rotation.toArray()),
    scale: rounded([scale, scale, scale]),
  };
}

// ── shared helpers ───────────────────────────────────────────────────────────

/** An artifact-producing stage: reused off disk when a previous run already made it. */
async function stage(file, make) {
  if (fs.existsSync(file)) return { bytes: fs.readFileSync(file), reused: true };
  const bytes = await make();
  writeAtomic(file, bytes);
  return { bytes, reused: false };
}

/** What `buildPrompt` wants for one object: its grid and the printable slices. */
const objectOf = (name, triangles, resolution) => {
  const grid = voxelize(triangles, resolution);
  return { name, grid, slices: toSlices(grid) };
};

const gridMeta = ({ dims, voxelSize, origin, center, size }) =>
  ({ dims, voxelSize: round(voxelSize), origin: rounded(origin), center: rounded(center), size: rounded(size) });

const readJson = (file) => JSON.parse(fs.readFileSync(file, 'utf8'));
const writeJson = (file, value) => writeAtomic(file, `${JSON.stringify(value, null, 2)}\n`);

// ── the generation half ──────────────────────────────────────────────────────

async function generateSample(id) {
  const say = (message) => console.log(`  [${id}] ${message}`);
  const { dir, anchorFile, imageFile, phrase } = readSample(SET_DIR, id);

  const out = path.join(OUT_DIR, id);
  // A new mesh invalidates every model result posed against the old one, so
  // forcing regeneration takes the whole sample folder with it.
  if (FORCE) fs.rmSync(out, { recursive: true, force: true });
  if (fs.existsSync(path.join(out, GENERATION_FILE))) {
    say('already generated — use --force to remesh');
    return;
  }
  fs.mkdirSync(out, { recursive: true });
  say(`"${phrase}"  anchor=${anchorFile}  object=${imageFile}`);

  const anchorBytes = fs.readFileSync(path.join(dir, anchorFile));
  const anchorCopy = path.join(out, 'anchor.glb');
  if (!fs.existsSync(anchorCopy)) writeAtomic(anchorCopy, anchorBytes);

  // 1 · the anchor as the image model will see it
  const view = await stage(path.join(out, 'anchor-view.png'), () =>
    renderView([{ triangles: sceneTriangles(parseGLB(anchorBytes)) }]));
  say(`1 · anchor view${view.reused ? ' (reused)' : ''}`);

  // 2 · the object re-posed as if placed, alone on white
  const generated = await stage(path.join(out, 'placed-generated.png'), () =>
    retry(() => generateImage({
      prompt: generationPrompt(phrase),
      model: IMAGE_MODEL,
      images: [
        { mimeType: mimeOf(imageFile), data: fs.readFileSync(path.join(dir, imageFile)) },
        { mimeType: 'image/png', data: view.bytes },
      ],
    })));
  say(`2 · placed-object image${generated.reused ? ' (reused)' : ''}`);

  // 3 · image-to-3D
  const raw = await stage(path.join(out, 'placed-raw.glb'), () =>
    generateMesh({ image: generated.bytes, name: `${id}-placed.png`, log: (message) => say(`    ${message}`) }));
  say(`3 · mesh ${(raw.bytes.length / 1e6).toFixed(1)} MB${raw.reused ? ' (reused)' : ''}`);

  writeJson(path.join(out, GENERATION_FILE), {
    id,
    placement: phrase,
    created_at: new Date().toISOString(),
    view: ISO_VIEW,
    image_model: IMAGE_MODEL,
    anchor: { source: anchorFile, mesh: 'anchor.glb', view_image: 'anchor-view.png' },
    placed: { source_image: imageFile, generated_image: 'placed-generated.png', raw_mesh: 'placed-raw.glb' },
  });
}

/** Steps 1-3 for every sample at once. Returns the ids that now have a mesh. */
export async function generateSamples(ids = listSamples()) {
  console.log(`\ngeneration · ${ids.length} sample(s) → ${path.relative(ROOT, OUT_DIR)}`);
  try {
    console.log(`  trellis → ${JSON.stringify(await health())}`);
  } catch (err) {
    console.log(`  trellis health check failed (continuing): ${err.message}`);
  }

  const ok = await mapLimit(ids, GENERATE_WIDTH, async (id) => {
    try {
      await generateSample(id);
      return true;
    } catch (err) {
      console.error(`  [${id}] ✗ ${err.message}`);
      return false;
    }
  });

  const ready = ids.filter((_, i) => ok[i]);
  console.log(`\n${ready.length}/${ids.length} generated`);
  return ready;
}

// ── the placement half ───────────────────────────────────────────────────────

/**
 * Everything a sample's models share: the two meshes and their occupancy grids.
 * Voxelizing is identical for every model and the slices are most of the prompt,
 * so it happens once per sample rather than once per call.
 */
function prepare(id) {
  const out = path.join(OUT_DIR, id);
  const marker = path.join(out, GENERATION_FILE);
  if (!fs.existsSync(marker)) throw new Error(`no ${GENERATION_FILE} — run pipeline/test-generate.mjs first`);
  const generation = readJson(marker);

  const anchorTriangles = sceneTriangles(parseGLB(fs.readFileSync(path.join(out, generation.anchor.mesh))));
  const placedGlb = forceOpaqueMaterials(parseGLB(fs.readFileSync(path.join(out, generation.placed.raw_mesh))));
  const placedTriangles = sceneTriangles(placedGlb);

  return {
    id,
    out,
    phrase: generation.placement,
    generation,
    anchorTriangles,
    placedTriangles,
    placedGlb,
    anchor: objectOf(nameOf(generation.anchor.source), anchorTriangles, RESOLUTION.anchor),
    placed: objectOf(nameOf(generation.placed.source_image), placedTriangles, RESOLUTION.placed),
  };
}

async function placeWithModel(sample, model) {
  const dir = path.join(sample.out, modelSlug(model));
  const marker = path.join(dir, PLACEMENT_FILE);
  if (FORCE) fs.rmSync(dir, { recursive: true, force: true });
  if (fs.existsSync(marker)) {
    const previous = readJson(marker);
    console.log(`  [${sample.id}] ${model} · already placed`);
    return { model, ok: true, latency: previous.latency_ms, cost: previous.usage?.cost ?? 0 };
  }
  fs.mkdirSync(dir, { recursive: true });

  const started = Date.now();
  const answer = await retry(() =>
    requestPlacement({ phrase: sample.phrase, a: sample.anchor, b: sample.placed, model }));
  const latency = Date.now() - started;

  const trs = resolveTransform(sample.anchor, sample.placed, answer);
  writeAtomic(path.join(dir, 'placed.glb'), serializeGLB(bakeTransform(sample.placedGlb, trs)));

  // The anchor is ghosted so a placement inside a cavity still shows.
  writeAtomic(path.join(dir, 'scene-view.png'), await renderView([
    { triangles: sample.anchorTriangles, dither: true },
    { triangles: transformTriangles(sample.placedTriangles, trs), base: 140 },
  ]));

  const { reasoning, scale, yaw_degrees, pitch_degrees, roll_degrees, center } = answer;
  writeJson(marker, {
    id: sample.id,
    model,
    model_used: answer.used,
    placement: sample.phrase,
    created_at: new Date().toISOString(),
    latency_ms: latency,
    anchor: { mesh: `../${sample.generation.anchor.mesh}`, grid: gridMeta(sample.anchor.grid) },
    placed: { mesh: 'placed.glb', raw_mesh: `../${sample.generation.placed.raw_mesh}`, grid: gridMeta(sample.placed.grid) },
    answer: { reasoning, scale, yaw_degrees, pitch_degrees, roll_degrees, center },
    transform: trs,
    scene_image: 'scene-view.png',
    usage: answer.usage,
  });

  console.log(
    `  [${sample.id}] ${model} · scale=${round(scale)} ypr=${yaw_degrees}/${pitch_degrees}/${roll_degrees}` +
    ` center=[${rounded(center)}]  ${(latency / 1000).toFixed(1)}s`,
  );
  return { model, ok: true, latency, cost: answer.usage?.cost ?? 0 };
}

/** One line per model: how many samples it placed, how long it took, what it cost. */
function report(results) {
  const width = Math.max(...MODELS.map((model) => model.length));
  console.log('');
  for (const model of MODELS) {
    const mine = results.filter((result) => result.model === model);
    const ok = mine.filter((result) => result.ok);
    const latencies = ok.map((result) => result.latency).filter(Boolean);
    const cost = ok.reduce((total, result) => total + (result.cost ?? 0), 0);
    const average = latencies.length ? latencies.reduce((a, b) => a + b, 0) / latencies.length / 1000 : 0;
    console.log(
      `  ${model.padEnd(width)}  ${ok.length}/${mine.length} placed` +
      `${average ? `  ${average.toFixed(1)}s avg` : ''}${cost ? `  $${cost.toFixed(4)}` : ''}`,
    );
  }
}

/** Steps 4-7 for every sample against every model, all in flight together. */
export async function placeSamples(ids = listSamples()) {
  console.log(`\nplacement · ${ids.length} sample(s) × ${MODELS.length} model(s)`);

  const prepared = [];
  for (const id of ids) {
    try {
      const sample = prepare(id);
      prepared.push(sample);
      console.log(`  [${id}] anchor ${sample.anchor.grid.dims.join('x')}  placed ${sample.placed.grid.dims.join('x')}`);
    } catch (err) {
      console.error(`  [${id}] ✗ ${err.message}`);
    }
  }

  const tasks = prepared.flatMap((sample) => MODELS.map((model) => ({ sample, model })));
  const results = await mapLimit(tasks, PLACE_WIDTH, async ({ sample, model }) => {
    try {
      return await placeWithModel(sample, model);
    } catch (err) {
      console.error(`  [${sample.id}] ${model} ✗ ${err.message}`);
      return { model, ok: false };
    }
  });

  report(results);
  return results.filter((result) => !result.ok).length;
}

export const sampleIds = () => {
  const requested = args.filter((arg) => !arg.startsWith('--'));
  return requested.length ? requested : listSamples();
};

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const failed = await placeSamples(await generateSamples(sampleIds()));
  if (failed) process.exitCode = 1;
}
