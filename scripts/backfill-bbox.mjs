// Backfill existing dataset samples with B's bounding box (b_only) and the
// A-to-B dimension ratio (a_to_b), matching what the live exporter now writes.
//
// Usage:  node scripts/backfill-bbox.mjs   (optionally set DATASET_DIR)
//
// B's bounding box is recomputed from each sample's exported B .glb by
// transforming every mesh's POSITION bounds by its node world matrix — the
// same result as THREE.Box3().setFromObject(objB) used in the app.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Matrix4, Vector3, Quaternion, Box3 } from 'three';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATASET_DIR = process.env.DATASET_DIR
  ? path.resolve(process.env.DATASET_DIR)
  : path.resolve(__dirname, '..', 'dataset');

const round = (v) => Number(v.toFixed(6));

function parseGLB(buf) {
  const totalLength = buf.readUInt32LE(8);
  let offset = 12;
  let json = null;
  let bin = null;
  while (offset < totalLength) {
    const chunkLength = buf.readUInt32LE(offset);
    const chunkType = buf.toString('ascii', offset + 4, offset + 8);
    const chunkData = buf.subarray(offset + 8, offset + 8 + chunkLength);
    if (chunkType.trim() === 'JSON') json = JSON.parse(chunkData.toString('utf8'));
    else if (chunkType.startsWith('BIN')) bin = chunkData;
    offset += 8 + chunkLength;
  }
  if (!json) throw new Error('no JSON chunk');
  return { json, bin };
}

const NUM_COMPONENTS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

function accessorBounds(json, bin, index) {
  const acc = json.accessors[index];
  if (acc.min && acc.max) return { min: acc.min, max: acc.max };
  // Fallback: derive bounds from the binary buffer (assumes float components).
  const view = json.bufferViews[acc.bufferView];
  const comps = NUM_COMPONENTS[acc.type];
  const base = (acc.byteOffset ?? 0) + (view.byteOffset ?? 0);
  const stride = view.byteStride ?? comps * 4;
  const min = new Array(comps).fill(Infinity);
  const max = new Array(comps).fill(-Infinity);
  for (let i = 0; i < acc.count; i++) {
    for (let c = 0; c < comps; c++) {
      const value = bin.readFloatLE(base + i * stride + c * 4);
      if (value < min[c]) min[c] = value;
      if (value > max[c]) max[c] = value;
    }
  }
  return { min, max };
}

function nodeLocalMatrix(node) {
  const m = new Matrix4();
  if (node.matrix) return m.fromArray(node.matrix);
  const t = node.translation ?? [0, 0, 0];
  const r = node.rotation ?? [0, 0, 0, 1];
  const s = node.scale ?? [1, 1, 1];
  return m.compose(new Vector3(...t), new Quaternion(...r), new Vector3(...s));
}

function worldSizeFromGLB(buf) {
  const { json, bin } = parseGLB(buf);
  const box = new Box3();
  const sceneIndex = json.scene ?? 0;
  const roots = json.scenes?.[sceneIndex]?.nodes ?? [];

  const visit = (nodeIndex, parentMatrix) => {
    const node = json.nodes[nodeIndex];
    const world = parentMatrix.clone().multiply(nodeLocalMatrix(node));
    if (node.mesh !== undefined) {
      for (const prim of json.meshes[node.mesh].primitives) {
        const posIndex = prim.attributes?.POSITION;
        if (posIndex === undefined) continue;
        const { min, max } = accessorBounds(json, bin, posIndex);
        for (const x of [min[0], max[0]])
          for (const y of [min[1], max[1]])
            for (const z of [min[2], max[2]])
              box.expandByPoint(new Vector3(x, y, z).applyMatrix4(world));
      }
    }
    for (const child of node.children ?? []) visit(child, world);
  };

  for (const root of roots) visit(root, new Matrix4());
  return box.getSize(new Vector3());
}

function boxInfoFromSize(size) {
  const max = Math.max(size.x, size.y, size.z, 1e-9);
  return {
    dimensions: { x: round(size.x), y: round(size.y), z: round(size.z) },
    ratios: { x: round(size.x / max), y: round(size.y / max), z: round(size.z / max) },
  };
}

function ratio(aDims, size) {
  const r = (n, d) => round(n / Math.max(d, 1e-9));
  return { x: r(aDims.x, size.x), y: r(aDims.y, size.y), z: r(aDims.z, size.z) };
}

function processSample(dir) {
  const metaPath = path.join(dir, 'metadata.json');
  if (!fs.existsSync(metaPath)) return { status: 'skip', reason: 'no metadata.json' };

  const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
  const aOnly = meta.bounding_box?.a_only;
  const bFile = meta.files?.b;
  if (!aOnly?.dimensions) return { status: 'skip', reason: 'no bounding_box.a_only' };
  if (!bFile) return { status: 'skip', reason: 'no files.b' };

  const bPath = path.join(dir, bFile);
  if (!fs.existsSync(bPath)) return { status: 'skip', reason: `missing ${bFile}` };

  const size = worldSizeFromGLB(fs.readFileSync(bPath));
  const bOnly = boxInfoFromSize(size);
  const aToB = ratio(aOnly.dimensions, size);

  meta.bounding_box = {
    a_only: aOnly,
    b_only: bOnly,
    a_with_b: meta.bounding_box.a_with_b,
    a_to_b: aToB,
  };

  fs.writeFileSync(metaPath, JSON.stringify(meta, null, 2) + '\n', 'utf8');
  return { status: 'ok', bOnly, aToB };
}

function main() {
  if (!fs.existsSync(DATASET_DIR)) {
    console.error(`Dataset directory not found: ${DATASET_DIR}`);
    process.exit(1);
  }

  const dirs = fs
    .readdirSync(DATASET_DIR, { withFileTypes: true })
    .filter((e) => e.isDirectory())
    .map((e) => path.join(DATASET_DIR, e.name));

  let updated = 0;
  let skipped = 0;
  let failed = 0;

  for (const dir of dirs) {
    const name = path.basename(dir);
    try {
      const result = processSample(dir);
      if (result.status === 'ok') {
        updated++;
        const d = result.bOnly.dimensions;
        console.log(`✓ ${name}  b=[${d.x}, ${d.y}, ${d.z}]  a/b=[${result.aToB.x}, ${result.aToB.y}, ${result.aToB.z}]`);
      } else {
        skipped++;
        console.log(`– ${name}  (${result.reason})`);
      }
    } catch (err) {
      failed++;
      console.error(`✗ ${name}  ${err.message}`);
    }
  }

  console.log(`\nDone. Updated ${updated}, skipped ${skipped}, failed ${failed} (dataset: ${DATASET_DIR}).`);
}

main();
