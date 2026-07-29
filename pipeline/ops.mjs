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
import { parseGLB, serializeGLB, sceneTriangles, bakeTransform, forceOpaqueMaterials } from './glb.mjs';
import { voxelize, toSlices } from './voxelize.mjs';
import { writeAtomic } from './metadata.mjs';

const FARM_OUT = process.env.FARM_OUT_DIR ?? '/farm-out';
const FARM_IN = process.env.FARM_IN_DIR ?? '/farm-in';
const SCENE = process.env.SCENE_DIR ?? '/scene';
const WORK = process.env.SCENE_WORK_PREFIX ?? 'datasets/raw/stage1-work';
const PUBLISH = process.env.SCENE_PUBLISH_PREFIX ?? 'datasets/raw/stage1';

const IMAGE_EXT = /\.(png|jpe?g|webp)$/i;

// Exit code for an input that is not on the volume, which is the one failure a caller can act
// on: it means their snapshot predates the write, and a reload fixes it. `modal/scene_ops.py`
// reloads and retries once on this code, so the two have to agree.
export const MISSING_INPUT = 17;

const workDir = (sample) => path.join(SCENE, WORK, sample);
const publishDir = (sample) => path.join(SCENE, PUBLISH, sample);
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

export const COMMANDS = {
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

  /** The occupancy grids the placement model reasons over. */
  voxelize({ sample, objects }) {
    const dir = workDir(sample);
    const grids = {};
    for (const { role, stem, resolution } of objects) {
      const grid = voxelize(sceneTriangles(readGlb(path.join(dir, `${stem}.glb`))), resolution);
      grids[role] = {
        dims: grid.dims,
        voxelSize: grid.voxelSize,
        origin: grid.origin,
        center: grid.center,
        size: grid.size,
        slices: toSlices(grid),
      };
    }
    return grids;
  },

  /**
   * Writes the posed meshes and their images into the published folder.
   *
   * The raw meshes stay untouched in the work folder, so this is idempotent and re-placing a
   * sample costs nothing — the old scheme baked in place, which meant a second pose would
   * have stacked on the first and a changed mind meant re-meshing on the farm.
   */
  bake({ sample, objects }) {
    const source = workDir(sample);
    const destination = publishDir(sample);
    fs.mkdirSync(destination, { recursive: true });

    const boxes = {};
    const sizes = {};
    for (const { role, stem, trs } of objects) {
      const glb = readGlb(path.join(source, `${stem}.glb`));
      boxes[role] = measure(glb, trs);
      writeAtomic(path.join(destination, `${stem}.glb`), serializeGLB(bakeTransform(glb, trs)));
      sizes[role] = rounded(boxes[role].max.map((value, c) => value - boxes[role].min[c]));

      const image = findByStem(source, stem, IMAGE_EXT);
      if (image) writeAtomic(path.join(destination, image), fs.readFileSync(path.join(source, image)));
    }

    const all = Object.values(boxes);
    return {
      sizes,
      combined_size: rounded(
        [0, 1, 2].map((c) => Math.max(...all.map((b) => b.max[c])) - Math.min(...all.map((b) => b.min[c]))),
      ),
    };
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
    process.exit(err.code === 'ENOENT' ? MISSING_INPUT : 1);
  }
}
