// The pool path end to end: two GLBs written to a temp folder, refined through the
// worker pool exactly the way a local corpus or the Modal pipeline would — GLB parsing,
// worker messaging, and a clean process exit (unref) all on the line at once.
//
//   node scripts/test-physics-pool.mjs

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { serializeGLB } from '../pipeline/glb.mjs';
import { refine } from '../pipeline/physics-pool.mjs';

const DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', '.tmp-physics-pool');

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

let failures = 0;
const check = (label, ok, detail = '') => {
  if (!ok) failures++;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${label}${detail ? `  (${detail})` : ''}`);
};

fs.rmSync(DIR, { recursive: true, force: true });
fs.mkdirSync(DIR, { recursive: true });
fs.writeFileSync(path.join(DIR, 'anchor.glb'), glbOf(box(0, 0, 0, 1, 1, 1)));
fs.writeFileSync(path.join(DIR, 'thing.glb'), glbOf(box(0, 0, 0, 0.1, 0.1, 0.1)));

const payload = (position, contact, direction) => ({
  objects: [
    { role: 'anchor', stem: 'anchor', trs: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] } },
    { role: 'placed', stem: 'thing', trs: { position, rotation: [0, 0, 0, 1], scale: [1, 1, 1] } },
  ],
  intent: { contact, direction, embed_fraction: 0 },
  options: {},
});

console.log('-- pool: two concurrent refines through workers');
const [mounted, rested] = await Promise.all([
  refine(DIR, payload([0.59, 0.2, 0], 'attach', [-1, 0, 0])),
  refine(DIR, payload([0, 0.58, 0], 'rest', [0, 0, 0])),
]);

check('attach met the wall', Math.abs(mounted.placed.position[0] - 0.548) < 0.008, mounted.placed.position[0].toFixed(4));
check('attach reported its contact', mounted.report.contact === 'attach');
check('rest met the top', Math.abs(rested.placed.position[1] - 0.548) < 0.008, rested.placed.position[1].toFixed(4));
check('rest kept its footing', Math.abs(rested.placed.position[0]) < 0.01, rested.placed.position[0].toFixed(4));

fs.rmSync(DIR, { recursive: true, force: true });
console.log(failures ? `${failures} check(s) FAILED` : 'all checks passed');
process.exitCode = failures ? 1 : 0;
// No explicit worker teardown: idle workers are unref'd, so reaching the end of this
// script must be enough for the process to exit. Hanging here IS a test failure.
