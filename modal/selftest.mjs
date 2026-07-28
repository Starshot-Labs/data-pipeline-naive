// End-to-end check of the Modal pipeline with the farm cut out.
//
//   node modal/selftest.mjs <step> [args]
//
// Driven step by step from `selftest.py`, which syncs the volume between them — the same
// boundary structure production uses, because the boundaries are half of what is being
// tested. Everything runs against its own prefixes so nothing it makes can reach the dataset.
//
// IT MUST NOT START A TRELLIS CAMPAIGN, and three separate things stop it:
//
//   1. it never imports `generateMeshes`, the only thing that calls `POST /run`. `buildImages`
//      is imported instead of `buildSamples`, so the mesh stage is not reachable from here.
//   2. `selftest.py` points `TRELLIS_BASE_URL` at a discard port, so a farm request could not
//      succeed even if some future edit tried to make one.
//   3. it stages images onto the farm's input volume and stops. Writing there costs nothing
//      and starts nothing; only `/run` does, and nothing here calls it.
//
// Meshes come from a sample the farm built earlier. Real ones, because voxelize needs actual
// triangles and the placement model needs a plausible shape — but they do not match the
// images beside them, which is exactly why this publishes into a throwaway prefix.

import fs from 'node:fs';
import path from 'node:path';
import { buildImages, GENERATED_DIR } from '../pipeline/build.mjs';
import { placeSamples } from '../pipeline/run.mjs';
import { uploadSamples } from '../pipeline/upload.mjs';
import { parseGLB, isBaked } from '../pipeline/glb.mjs';
import * as scene from '../pipeline/scene.mjs';
import * as meta from '../pipeline/metadata.mjs';

const SCENE = process.env.SCENE_DIR ?? '/scene';
const WORK = path.join(SCENE, process.env.SCENE_WORK_PREFIX ?? 'datasets/raw/selftest-work');
const PUBLISHED = path.join(SCENE, process.env.SCENE_PUBLISH_PREFIX ?? 'datasets/raw/selftest');
const DONOR_ROOT = process.env.SELFTEST_DONOR_ROOT ?? path.join(SCENE, 'datasets/raw/stage1-work');
const RUN_FILE = path.join(GENERATED_DIR, '.selftest-run');

const LANE_OF_ROLE = { anchor: 'geo', placed: 'tex' };

let failures = 0;
function check(label, ok, detail = '') {
  if (!ok) failures++;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${label}${detail ? `  (${detail})` : ''}`);
}

const samples = () => meta.list(GENERATED_DIR);
/** Only the samples stage 2 actually finished. A render that failed leaves one behind, and
 *  every later step would otherwise trip over its missing file rather than report it. */
const ready = () => samples().filter((sample) => meta.ROLES.every((role) => sample.metadata[role].image));

/** An anchor/placed mesh pair the farm produced earlier, to stand in for a campaign. */
function donor() {
  for (const name of fs.existsSync(DONOR_ROOT) ? fs.readdirSync(DONOR_ROOT) : []) {
    const dir = path.join(DONOR_ROOT, name);
    const glbs = fs.statSync(dir).isDirectory() ? fs.readdirSync(dir).filter((f) => f.endsWith('.glb')) : [];
    // The anchor's stem is the sample id, so the other file is the placed object.
    if (glbs.length === 2) {
      const anchor = glbs.find((f) => f === `${name}.glb`) ?? glbs[0];
      return { anchor: path.join(dir, anchor), placed: path.join(dir, glbs.find((f) => f !== anchor)) };
    }
  }
  throw new Error(`no donor mesh pair under ${DONOR_ROOT}`);
}

const STEPS = {
  /** Stages 1-2 for real: OpenRouter invents, Google renders, both through the secret. */
  async images(pairs) {
    console.log(`  farm base url: ${process.env.TRELLIS_BASE_URL}  (unreachable on purpose)`);
    const ready = await buildImages({ pairs: Number(pairs) });
    check('stage 1-2 produced samples with both images', ready.length > 0, `${ready.length} of ${pairs} asked`);
    for (const sample of samples()) {
      const stem = meta.stems(sample.metadata);
      for (const role of meta.ROLES) {
        const file = path.join(sample.dir, meta.imageName(stem[role]));
        const png = fs.existsSync(file) && fs.readFileSync(file).subarray(1, 4).toString('ascii') === 'PNG';
        check(`${sample.id} ${role} image is a PNG`, png);
      }
    }
  },

  /**
   * Staging only. This puts the images on the farm's input volume, which is where a campaign
   * would read them from, and then stops — the point being that `dc-scene-ops` can see files
   * the pipeline container wrote, which is only true because the volume was synced first.
   */
  async stage() {
    const runId = `selftest-${Date.now().toString(36)}`;
    const objects = [];
    for (const [index, sample] of ready().entries()) {
      const stem = meta.stems(sample.metadata);
      for (const role of meta.ROLES) {
        objects.push({
          source: path.join(sample.dir, meta.imageName(stem[role])),
          key: `o${index * 2 + (role === 'anchor' ? 0 : 1)}`,
          lane: LANE_OF_ROLE[role],
        });
      }
    }

    const { dirs, counts } = await scene.stageImages(runId, objects);
    meta.writeAtomic(RUN_FILE, runId);
    check('staged both lanes onto the farm input volume', Boolean(counts.geo && counts.tex), JSON.stringify(counts));
    check('returned the lane dirs /run would want', Boolean(dirs.texture_dir && dirs.geometry_dir), JSON.stringify(dirs));
    console.log(`  farm-in dirs to delete afterwards: uploads/${runId}-geo, uploads/${runId}-tex`);
  },

  /** What `collect` would have written, had a campaign run: meshes plus their images. */
  async inject() {
    const meshes = donor();
    console.log(`  donor: ${path.basename(path.dirname(meshes.anchor))}`);
    for (const sample of ready()) {
      const stem = meta.stems(sample.metadata);
      const target = path.join(WORK, sample.id);
      fs.mkdirSync(target, { recursive: true });
      for (const role of meta.ROLES) {
        meta.writeAtomic(path.join(target, `${stem[role]}.glb`), fs.readFileSync(meshes[role]));
        const image = meta.imageName(stem[role]);
        meta.writeAtomic(path.join(target, image), fs.readFileSync(path.join(sample.dir, image)));
        meta.recordMesh(sample.metadata, role, `${stem[role]}.glb`);
      }
      meta.write(sample.dir, sample.metadata);
      check(`${sample.id} recorded both meshes`, meta.isMeshed(sample.metadata));
    }
  },

  /** Stages 4-6: voxelize and bake in-process on the mounted volume, place for real. */
  async place() {
    const ids = samples().filter((sample) => meta.isMeshed(sample.metadata)).map((sample) => sample.id);
    check('there are meshed samples to place', ids.length > 0, `${ids.length}`);
    const failed = await placeSamples({ sourceDir: GENERATED_DIR, ids });
    check('every meshed sample placed', failed === 0, `${failed} failed of ${ids.length}`);
  },

  /** Stage 7. */
  async publish() {
    const expected = samples().filter((sample) => meta.isPlaced(sample.metadata)).length;
    const done = await uploadSamples();
    check('published every placed sample', done === expected && expected > 0, `${done} of ${expected}`);
  },

  /** What the dataset should now look like. */
  async verify() {
    const up = await scene.published();
    const placed = samples().filter((sample) => meta.isPlaced(sample.metadata));
    check('there are placed samples to verify', placed.length > 0, `${placed.length}`);
    for (const sample of placed) {
      const { id, metadata } = sample;
      const stem = meta.stems(metadata);

      const sizes = [metadata.anchor.size, metadata.placed.size, metadata.combined_size];
      check(`${id} has all three size triples`, sizes.every((s) => Array.isArray(s) && s.length === 3 && s.every(Number.isFinite)));
      // The union box contains both, so on every axis it is at least as wide as either.
      if (sizes.every(Array.isArray)) {
        const encloses = [0, 1, 2].every((c) => metadata.combined_size[c] >= Math.max(metadata.anchor.size[c], metadata.placed.size[c]) - 1e-6);
        check(`${id} combined_size encloses both`, encloses, JSON.stringify(metadata.combined_size));
      }

      const dir = path.join(PUBLISHED, id);
      const files = fs.existsSync(dir) ? fs.readdirSync(dir) : [];
      const wanted = ['metadata.json', 'placement.txt', ...meta.ROLES.flatMap((r) => [`${stem[r]}.glb`, meta.imageName(stem[r])])];
      check(`${id} published all six files`, wanted.every((f) => files.includes(f)), `${files.length} present`);

      for (const role of meta.ROLES) {
        const glb = path.join(dir, `${stem[role]}.glb`);
        const posed = fs.existsSync(glb) && isBaked(parseGLB(fs.readFileSync(glb)));
        check(`${id} ${role} mesh carries a baked pose`, posed);
      }
      check(`${id} is listed as published`, up.has(id));
    }
  },

  /** Everything this test made, gone. The farm-in dirs are the caller's to remove. */
  async clean() {
    for (const directory of [GENERATED_DIR, WORK, PUBLISHED]) {
      fs.rmSync(directory, { recursive: true, force: true });
      console.log(`  removed ${directory}`);
    }
  },
};

const [step, ...args] = process.argv.slice(2);
if (!STEPS[step]) {
  console.error(`unknown step "${step}" — expected one of ${Object.keys(STEPS).join(', ')}`);
  process.exit(2);
}
console.log(`-- ${step}`);
await STEPS[step](...args);
if (failures) {
  console.error(`  ${failures} check(s) failed`);
  process.exit(1);
}
