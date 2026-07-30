// Every operation that needs the volumes mounted.
//
// Reached two ways, and the caller does not care which. `scene.mjs` imports `COMMANDS` and
// runs them in-process when the pipeline itself is on Modal; `modal/scene_ops.py` shells out
// to this file per request when the pipeline is driving from a laptop:
//
//   node pipeline/ops.mjs <command>   JSON request on stdin, JSON reply on stdout
//
// Meshes are big and everything that touches them — collecting them off the farm, voxelizing
// them, baking a pose into them — is cheap. Doing that beside the volumes rather than at home
// is worth roughly 13 MB a sample of round trip, which is why this file exists.
//
// Three volumes:
//   /farm-out   the farm's output, where campaigns leave their GLBs and image shards
//   /farm-in    the farm's input, where the images we send it live
//   /scene      ours: raw samples under WORK, finished ones under PUBLISH

import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import sharp from 'sharp';
import { decompress } from 'fzstd';
import { Vector3, Quaternion, Matrix4 } from 'three';
import { parseGLB, serializeGLB, sceneTriangles, bakeTransform, deformGLB, extractTransform, forceOpaqueMaterials, isBaked } from './glb.mjs';
import { voxelize, toBlocks } from './voxelize.mjs';
import { refineDir } from './physics.mjs';
import { buildDrape } from './cloth.mjs';
import { renderGLB } from './render.mjs';
import { writeAtomic } from './metadata.mjs';

const FARM_OUT = process.env.FARM_OUT_DIR ?? '/farm-out';
const FARM_IN = process.env.FARM_IN_DIR ?? '/farm-in';
const SCENE = process.env.SCENE_DIR ?? '/scene';
const WORK = process.env.SCENE_WORK_PREFIX ?? 'datasets/raw/stage1-work';
const PUBLISH = process.env.SCENE_PUBLISH_PREFIX ?? 'datasets/raw/stage1';
// Objaverse GLBs and their renders, cached by uid — assets shared across samples land once.
const CACHE = process.env.SCENE_CACHE_PREFIX ?? 'datasets/raw/objaverse-cache';

const IMAGE_EXT = /\.(png|jpe?g|webp)$/i;

export const workDir = (sample) => path.join(SCENE, WORK, sample);
export const publishDir = (sample) => path.join(SCENE, PUBLISH, sample);
const round = (v) => Number(v.toFixed(6));
const rounded = (values) => values.map(round);

const listDir = (dir) => (fs.existsSync(dir) ? fs.readdirSync(dir) : []);
const findByStem = (dir, stem, pattern) =>
  listDir(dir).find((file) => file.startsWith(`${stem}.`) && pattern.test(file)) ?? null;

/**
 * The farm names a per-object output `<folder>__<...>__<key>` while the images we upload are
 * named plainly `<key>`. Taking the tail after `__` accepts both, so this depends on the key
 * alone rather than on a prefix convention that is the farm's to change.
 */
const findByKey = (dir, key, pattern) =>
  listDir(dir).find(
    (file) => pattern.test(file) && path.basename(file, path.extname(file)).split('__').pop() === key,
  ) ?? null;

/**
 * The conditioning images of one shard, as `key → complete PNG bytes`.
 *
 * A shard is zstd-compressed safetensors: a u64 header length, that much JSON, then the
 * tensor data. Its `_meta` is a JSON blob smuggled in as a uint8 tensor, `offsets` is a
 * prefix sum, and every object's PNG is concatenated into one `images` array — so object
 * `k` is `images[offsets[k]:offsets[k+1]]`, already a complete file. Parsed by hand
 * because that is the whole format; a safetensors library would be more surface than the
 * dozen lines it replaces, and only the three tensors named here are ever wanted.
 */
function readImageShard(file) {
  const blob = Buffer.from(decompress(fs.readFileSync(file)));
  const headerLength = Number(blob.readBigUInt64LE(0));
  const header = JSON.parse(blob.subarray(8, 8 + headerLength).toString('utf8'));
  const base = 8 + headerLength;
  const tensor = (name) =>
    header[name] ? blob.subarray(base + header[name].data_offsets[0], base + header[name].data_offsets[1]) : null;

  const meta = tensor('_meta');
  const offsets = tensor('offsets');
  const images = tensor('images');
  if (!meta || !offsets || !images) return new Map();

  const ids = JSON.parse(meta.toString('utf8')).object_ids ?? [];
  const byKey = new Map();
  for (const [index, id] of ids.entries()) {
    const start = Number(offsets.readBigInt64LE(index * 8));
    const end = Number(offsets.readBigInt64LE((index + 1) * 8));
    byKey.set(String(id).split('__').pop(), images.subarray(start, end));
  }
  return byKey;
}

/**
 * Writes an image the dataset can rely on being PNG.
 *
 * Shard bytes go down untouched. The shards describe themselves as lossless and exact —
 * "re-encoding it reproduces this run's conditioning" — so passing them through an encoder
 * could only drift from the frames the meshes were actually built from.
 */
async function writeImage(source, destination) {
  const bytes = Buffer.isBuffer(source) ? source : fs.readFileSync(source);
  const png = bytes.subarray(1, 4).toString('ascii') === 'PNG' ? bytes : await sharp(bytes).png().toBuffer();
  writeAtomic(destination, png);
}

const readGlb = (file) => {
  const data = fs.readFileSync(file);
  if (data.subarray(0, 4).toString('ascii') !== 'glTF') throw new Error(`${path.basename(file)} is not a GLB`);
  return parseGLB(data);
};

/** The world-space box of a mesh once `trs` is applied, measured off the transformed
 *  triangles — a transformed bounding box only agrees when the yaw is a multiple of 90°. */
function measure(glb, trs) {
  const matrix = new Matrix4().compose(
    new Vector3().fromArray(trs.position),
    new Quaternion().fromArray(trs.rotation),
    new Vector3().fromArray(trs.scale),
  );
  const triangles = sceneTriangles(glb);
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  const vertex = new Vector3();
  for (let i = 0; i < triangles.length; i += 3) {
    vertex.set(triangles[i], triangles[i + 1], triangles[i + 2]).applyMatrix4(matrix);
    for (let c = 0; c < 3; c++) {
      const value = vertex.getComponent(c);
      if (value < min[c]) min[c] = value;
      if (value > max[c]) max[c] = value;
    }
  }
  return { min, max };
}

/**
 * Grid + solid-block decomposition for the meshes in `dir`: `objects` is
 * `{ role, stem, resolution }`. Exported apart from the command so a pipeline whose sample
 * folders already hold their meshes can voxelize them in-process — same code, same grids,
 * no volume in the loop.
 */
export function voxelizeDir(dir, objects) {
  const grids = {};
  for (const { role, stem, resolution } of objects) {
    const grid = voxelize(sceneTriangles(readGlb(path.join(dir, `${stem}.glb`))), resolution);
    grids[role] = {
      dims: grid.dims,
      voxelSize: grid.voxelSize,
      origin: grid.origin,
      center: grid.center,
      size: grid.size,
      blocks: toBlocks(grid),
    };
  }
  return grids;
}

/**
 * The placement TRS each of a sample's posed meshes carries, read back from the files —
 * the GLB is the one place a pose lives, so a pass that continues from the current
 * arrangement starts here. `name` maps a stem to its posed file: published folders keep
 * the plain name, a local corpus poses into `.posed.glb` siblings.
 */
export function posedTransforms(dir, objects, name = (stem) => `${stem}.glb`) {
  const transforms = {};
  for (const { role, stem } of objects) {
    const trs = extractTransform(readGlb(path.join(dir, name(stem))));
    if (!trs) throw new Error(`${name(stem)} carries no placement — place the sample first`);
    transforms[role] = trs;
  }
  return transforms;
}

/**
 * Reads each raw mesh from `source`, bakes its transform in, and writes the posed copy and
 * its reference image into `destination`. Exported apart from the command for local corpora,
 * which bake with `source` and `destination` the same folder: there the posed copy is
 * written as a `.posed.glb` sibling and the raw mesh survives untouched, so the sample can
 * be re-placed any number of times. A raw file already carrying a placement (a folder baked
 * by the old in-place scheme) is refused rather than given a second transform.
 */
export function bakeDir(source, destination, objects) {
  fs.mkdirSync(destination, { recursive: true });

  const boxes = {};
  const sizes = {};
  for (const { role, stem, trs } of objects) {
    const glb = readGlb(path.join(source, `${stem}.glb`));
    if (isBaked(glb)) {
      throw new Error(`${stem}.glb already carries a placement — re-placing needs the raw mesh back`);
    }
    boxes[role] = measure(glb, trs);
    const name = source === destination ? `${stem}.posed.glb` : `${stem}.glb`;
    writeAtomic(path.join(destination, name), serializeGLB(bakeTransform(glb, trs)));
    sizes[role] = rounded(boxes[role].max.map((value, c) => value - boxes[role].min[c]));

    const image = findByStem(source, stem, IMAGE_EXT);
    if (image && source !== destination) {
      writeAtomic(path.join(destination, image), fs.readFileSync(path.join(source, image)));
    }
  }

  const all = Object.values(boxes);
  return {
    sizes,
    combined_size: rounded(
      [0, 1, 2].map((c) => Math.max(...all.map((b) => b.max[c])) - Math.min(...all.map((b) => b.min[c]))),
    ),
  };
}

/**
 * The soft-body counterpart of `bakeDir`: reads both raw meshes, drapes the placed one
 * over the anchor (pipeline/cloth.mjs), and writes the result in one step — the
 * intermediate is a whole vertex buffer, which is not worth a second trip. The deformed
 * positions are written back into the placed GLB in its own local frame under the same
 * placement TRS, so the file behaves exactly like a rigidly-baked one to everything
 * downstream. A drape that refuses falls back to the rigid bake of the model's answer,
 * with the report saying so.
 */
export function drapeDir(source, destination, objects, options = {}) {
  fs.mkdirSync(destination, { recursive: true });
  const byRole = Object.fromEntries(objects.map((object) => [object.role, object]));
  const glbs = {};
  for (const role of ['anchor', 'placed']) {
    glbs[role] = readGlb(path.join(source, `${byRole[role].stem}.glb`));
    if (isBaked(glbs[role])) {
      throw new Error(`${byRole[role].stem}.glb already carries a placement — re-placing needs the raw mesh back`);
    }
  }

  const { map, report } = buildDrape({
    anchorTriangles: sceneTriangles(glbs.anchor),
    placedTriangles: sceneTriangles(glbs.placed),
    anchor: byRole.anchor.trs,
    placed: byRole.placed.trs,
    options,
  });
  if (map) glbs.placed = deformGLB(glbs.placed, map);

  const boxes = {};
  const sizes = {};
  for (const { role, stem, trs } of objects) {
    boxes[role] = measure(glbs[role], trs);
    sizes[role] = rounded(boxes[role].max.map((value, c) => value - boxes[role].min[c]));
    const name = source === destination ? `${stem}.posed.glb` : `${stem}.glb`;
    writeAtomic(path.join(destination, name), serializeGLB(bakeTransform(glbs[role], trs)));

    const image = findByStem(source, stem, IMAGE_EXT);
    if (image && source !== destination) {
      writeAtomic(path.join(destination, image), fs.readFileSync(path.join(source, image)));
    }
  }

  const all = Object.values(boxes);
  return {
    sizes,
    combined_size: rounded(
      [0, 1, 2].map((c) => Math.max(...all.map((b) => b.max[c])) - Math.min(...all.map((b) => b.min[c]))),
    ),
    report,
  };
}

const HF_GLB = (glbPath) => `https://huggingface.co/datasets/allenai/objaverse/resolve/main/${glbPath}`;

async function downloadTo(file, url) {
  for (let attempt = 1; ; attempt++) {
    try {
      const response = await fetch(url, { redirect: 'follow' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      writeAtomic(file, Buffer.from(await response.arrayBuffer()));
      return;
    } catch (err) {
      if (attempt >= 3) throw new Error(`${url}: ${err.message}`);
      await new Promise((resolve) => setTimeout(resolve, 2 ** attempt * 1000 * (0.5 + Math.random())));
    }
  }
}

/**
 * Downloads each object's Objaverse GLB and renders its reference image into
 * `destination`, both through a per-uid cache: an asset seeded into fifty samples is
 * downloaded and rendered once and copied fifty times. Exported apart from the command so
 * a local corpus can fetch straight into its own folder with a cache beside it — same
 * code, no volume in the loop. Failures come back per object rather than throwing, so one
 * bad asset (a Draco mesh, a dead file) does not block the sample's other half.
 */
export async function fetchInto(destination, objects, cacheDir) {
  fs.mkdirSync(destination, { recursive: true });
  fs.mkdirSync(cacheDir, { recursive: true });

  const fetched = {};
  const errors = [];
  for (const { role, stem, uid, glb } of objects) {
    try {
      const cachedGlb = path.join(cacheDir, `${uid}.glb`);
      if (!fs.existsSync(cachedGlb)) await downloadTo(cachedGlb, HF_GLB(glb));

      const cachedPng = path.join(cacheDir, `${uid}.png`);
      if (!fs.existsSync(cachedPng)) writeAtomic(cachedPng, await renderGLB(readGlb(cachedGlb)));

      writeAtomic(path.join(destination, `${stem}.glb`), fs.readFileSync(cachedGlb));
      writeAtomic(path.join(destination, `${stem}.png`), fs.readFileSync(cachedPng));
      fetched[role] = { mesh: `${stem}.glb`, image: `${stem}.png` };
    } catch (err) {
      errors.push({ role, uid, error: err.message });
    }
  }
  return { fetched, errors };
}

export const COMMANDS = {
  /**
   * A sample's recorded Objaverse assets, downloaded from Hugging Face onto the volume and
   * rendered, cache-first. `objects` is `[{ role, stem, uid, glb }]`.
   */
  fetch({ sample, objects }) {
    return fetchInto(workDir(sample), objects, path.join(SCENE, CACHE));
  },

  /**
   * Puts reference images where the farm's dispatcher looks: one directory per lane under
   * `uploads/`, each file named after its key so the object id the farm derives maps straight
   * back to a job. Only reachable in-process — over HTTP the service cannot see the caller's
   * files, so `scene.mjs` uses the Modal CLI for that case instead.
   */
  stage({ run_id, objects }) {
    const counts = {};
    for (const { source, key, lane } of objects) {
      const destination = path.join(FARM_IN, 'uploads', `${run_id}-${lane}`);
      fs.mkdirSync(destination, { recursive: true });
      writeAtomic(path.join(destination, `${key}${path.extname(source)}`), fs.readFileSync(source));
      counts[lane] = (counts[lane] ?? 0) + 1;
    }
    return { counts };
  },

  /** How many GLBs a campaign has written so far, which is what says it is finished. */
  assets({ run_id }) {
    return { count: listDir(path.join(FARM_OUT, run_id, 'glb')).filter((file) => /\.glb$/i.test(file)).length };
  },

  /**
   * Moves a finished campaign's meshes and their reference images onto our own volume, under
   * the names the sample uses. Every copy is volume-to-volume, so nothing crosses the
   * network. Meshes lose their `alphaMode: BLEND` here — bakers set it whenever a texture
   * atlas carries an alpha channel, and loaders answer it by dropping depth writes.
   */
  async collect({ run_id, objects }) {
    const glbDir = path.join(FARM_OUT, run_id, 'glb');
    const missing = [];
    const landed = [];

    for (const object of objects) {
      const source = findByKey(glbDir, object.key, /\.glb$/i);
      if (!source) {
        missing.push({ id: object.sample, role: object.role, stem: object.stem, key: object.key });
        continue;
      }
      const destination = workDir(object.sample);
      fs.mkdirSync(destination, { recursive: true });
      writeAtomic(
        path.join(destination, `${object.stem}.glb`),
        serializeGLB(forceOpaqueMaterials(readGlb(path.join(glbDir, source)))),
      );
      landed.push(object);
    }

    // Walk the shards rather than the objects: each one is a single zstd frame holding a
    // whole group, so asking it per object would decompress the same megabytes sixteen
    // times over. Whatever a shard carries is written and struck off the list.
    const wanted = new Map(landed.map((object) => [object.key, object]));
    const withImage = new Set();
    const shardDir = path.join(FARM_OUT, run_id, 'images');
    for (const shard of listDir(shardDir).filter((file) => file.endsWith('.st.zst'))) {
      for (const [key, bytes] of readImageShard(path.join(shardDir, shard))) {
        const object = wanted.get(key);
        if (!object) continue;
        await writeImage(bytes, path.join(workDir(object.sample), `${object.stem}.png`));
        wanted.delete(key);
        withImage.add(key);
      }
    }

    // A run from before the farm wrote its conditioning images falls back to the photo we
    // sent up: the same subject, but uncropped and with its original background.
    const images = { normalized: withImage.size, reference: 0, none: 0 };
    for (const object of wanted.values()) {
      const sent = path.join(FARM_IN, 'uploads', `${run_id}-${object.lane}`);
      const reference = findByKey(sent, object.key, IMAGE_EXT);
      if (!reference) {
        images.none++;
        continue;
      }
      await writeImage(path.join(sent, reference), path.join(workDir(object.sample), `${object.stem}.png`));
      withImage.add(object.key);
      images.reference++;
    }

    const collected = landed.map((object) => ({
      id: object.sample,
      role: object.role,
      stem: object.stem,
      mesh: `${object.stem}.glb`,
      image: withImage.has(object.key) ? `${object.stem}.png` : null,
    }));
    return { collected, missing, images };
  },

  /** The solid-block decompositions the placement model reasons over. */
  voxelize({ sample, objects }) {
    return voxelizeDir(workDir(sample), objects);
  },

  /** Mesh-accurate placement refinement, between place and bake — see pipeline/physics.mjs. */
  refine({ sample, objects, intent, options }) {
    return refineDir(workDir(sample), { objects, intent, options });
  },

  /** The baked placement transforms of a published sample, for a physics-only re-pass. */
  pose({ sample, objects }) {
    return { transforms: posedTransforms(publishDir(sample), objects) };
  },

  /** Soft-body placement: drape, deform and bake in one step — see pipeline/cloth.mjs. */
  drape({ sample, objects, options }) {
    return drapeDir(workDir(sample), publishDir(sample), objects, options);
  },

  /**
   * Writes the posed meshes and their images into the published folder.
   *
   * The raw meshes stay untouched in the work folder, so this is idempotent and re-placing a
   * sample costs nothing — the old scheme baked in place, which meant a second pose would
   * have stacked on the first and a changed mind meant re-meshing on the farm.
   */
  bake({ sample, objects }) {
    return bakeDir(workDir(sample), publishDir(sample), objects);
  },

  /**
   * The small text files for a batch of samples, written last so their presence is what marks
   * each one complete.
   *
   * Batched because the writing is not the cost: a sample's two files are a couple of
   * kilobytes, while the round trip and the volume commit around them are most of a second.
   * One call pays that once for the whole batch rather than once per sample.
   */
  publish({ samples }) {
    const published = [];
    for (const { sample, files } of samples) {
      const destination = publishDir(sample);
      fs.mkdirSync(destination, { recursive: true });
      for (const [name, contents] of Object.entries(files)) {
        writeAtomic(path.join(destination, path.basename(name)), contents);
      }
      published.push(sample);
    }
    return { published };
  },

  /** Samples already carrying a metadata.json, which is what makes one finished. */
  published() {
    const root = path.join(SCENE, PUBLISH);
    return {
      ids: listDir(root).filter((name) => fs.existsSync(path.join(root, name, 'metadata.json'))),
    };
  },
};

// Only when run as a script. Imported, this file is just `COMMANDS`.
if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  const command = process.argv[2];
  if (!COMMANDS[command]) {
    process.stderr.write(`unknown command "${command}" — expected one of ${Object.keys(COMMANDS).join(', ')}\n`);
    process.exit(2);
  }

  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  const request = chunks.length ? JSON.parse(Buffer.concat(chunks).toString('utf8')) : {};

  try {
    process.stdout.write(JSON.stringify(await COMMANDS[command](request)));
  } catch (err) {
    process.stderr.write(err.stack ?? String(err));
    process.exit(1);
  }
}
