// Builds a self-contained local sample — synthetic meshes in the folder, pipeline-format
// metadata.json beside them — so stages 4-6 can be exercised end to end with no farm, no
// volume and no images. The anchor is a table (slab on four legs), the placed object a
// small crate; the phrase asks for a rest placement the physics pass must finish.
//
//   node scripts/make-test-sample.mjs [dir]     default dir: .tmp-local-sample
//
// Prints the sample id and the run command. Re-running replaces the sample.

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { serializeGLB } from '../pipeline/glb.mjs';
import { slug, stems, write } from '../pipeline/metadata.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SOURCE = path.resolve(ROOT, process.argv[2] ?? '.tmp-local-sample');

function box(cx, cy, cz, sx, sy, sz) {
  const x0 = cx - sx / 2, x1 = cx + sx / 2;
  const y0 = cy - sy / 2, y1 = cy + sy / 2;
  const z0 = cz - sz / 2, z1 = cz + sz / 2;
  const v = [
    [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
    [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
  ];
  const faces = [
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [3, 6, 2], [3, 7, 6],
    [0, 7, 3], [0, 4, 7], [1, 2, 6], [1, 6, 5],
  ];
  const out = [];
  for (const [a, b, c] of faces) out.push(...v[a], ...v[b], ...v[c]);
  return out;
}

function glbOf(triangles) {
  const positions = new Float32Array(triangles);
  const bin = Buffer.from(positions.buffer.slice(0));
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    for (let c = 0; c < 3; c++) {
      if (positions[i + c] < min[c]) min[c] = positions[i + c];
      if (positions[i + c] > max[c]) max[c] = positions[i + c];
    }
  }
  const json = {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
    accessors: [{ bufferView: 0, componentType: 5126, count: positions.length / 3, type: 'VEC3', min, max }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: bin.length }],
    buffers: [{ byteLength: bin.length }],
  };
  return serializeGLB({ json, bin });
}

// A table: slab on four corner legs. Leg cross-sections are fat enough to survive the
// stage-4 voxel grid, so the placement model sees the same structure the physics does.
const LEG = 0.07;
const table = [
  ...box(0, 0.475, 0, 1, 0.05, 0.7),
  ...box(-0.45, 0.225, -0.3, LEG, 0.45, LEG),
  ...box(0.45, 0.225, -0.3, LEG, 0.45, LEG),
  ...box(-0.45, 0.225, 0.3, LEG, 0.45, LEG),
  ...box(0.45, 0.225, 0.3, LEG, 0.45, LEG),
];
const crate = box(0, 0, 0, 0.1, 0.1, 0.1);

const uuid = crypto.randomUUID();
const metadata = {
  id: `${slug('test table')}_${uuid}`,
  uuid,
  created_at: new Date().toISOString(),
  context: 'local physics test',
  phrasing: 'both',
  placement: 'wooden crate resting on top of the test table, near the back left corner',
  anchor: {
    name: 'test table',
    description: 'A plain rectangular table with a flat top slab standing on four square legs.',
    textured: false,
  },
  placed: {
    name: 'wooden crate',
    description: 'A small plain wooden cube crate.',
    textured: true,
  },
};

const stem = stems(metadata);
metadata.anchor.mesh = `${stem.anchor}.glb`;
metadata.placed.mesh = `${stem.placed}.glb`;

const dir = path.join(SOURCE, metadata.id);
fs.rmSync(dir, { recursive: true, force: true });
fs.mkdirSync(dir, { recursive: true });
fs.writeFileSync(path.join(dir, metadata.anchor.mesh), glbOf(table));
fs.writeFileSync(path.join(dir, metadata.placed.mesh), glbOf(crate));
write(dir, metadata);
fs.writeFileSync(path.join(dir, 'placement.txt'), metadata.placement);

console.log(`sample ${metadata.id}`);
console.log(`  node pipeline/run.mjs --source=${path.relative(ROOT, SOURCE)} ${metadata.id}`);
