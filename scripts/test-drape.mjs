// Numerical checks for the soft-body drape, on synthetic meshes — a cloth sheet with
// real overhang must bend over the anchor while its supported middle stays put, and the
// deformed result must come back as a well-formed GLB under the unchanged placement TRS.
//
//   node scripts/test-drape.mjs

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildDrape } from '../pipeline/cloth.mjs';
import { drapeDir } from '../pipeline/ops.mjs';
import { parseGLB, sceneTriangles, serializeGLB, extractTransform } from '../pipeline/glb.mjs';

const TMP = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', '.tmp-drape-test');

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

/** A flat sheet in the xz plane: (nx+1)·(nz+1) vertices, two triangles per quad. */
function sheet(w, d, nx, nz) {
  const positions = [];
  for (let j = 0; j <= nz; j++)
    for (let i = 0; i <= nx; i++) positions.push((i / nx - 0.5) * w, 0, (j / nz - 0.5) * d);
  const indices = [];
  const at = (i, j) => j * (nx + 1) + i;
  for (let j = 0; j < nz; j++)
    for (let i = 0; i < nx; i++) {
      indices.push(at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j), at(i + 1, j + 1), at(i, j + 1));
    }
  return { positions: new Float64Array(positions), indices };
}

const sheetTriangles = ({ positions, indices }) => {
  const out = new Float64Array(indices.length * 3);
  for (const [k, index] of indices.entries()) {
    out[k * 3] = positions[index * 3];
    out[k * 3 + 1] = positions[index * 3 + 1];
    out[k * 3 + 2] = positions[index * 3 + 2];
  }
  return out;
};

function sheetGlb({ positions, indices }) {
  const pos = new Float32Array(positions);
  const idx = new Uint32Array(indices);
  const posBytes = Buffer.from(pos.buffer.slice(0));
  const idxBytes = Buffer.from(idx.buffer.slice(0));
  const bin = Buffer.concat([posBytes, idxBytes]);
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < pos.length; i += 3)
    for (let c = 0; c < 3; c++) {
      if (pos[i + c] < min[c]) min[c] = pos[i + c];
      if (pos[i + c] > max[c]) max[c] = pos[i + c];
    }
  const json = {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: pos.length / 3, type: 'VEC3', min, max },
      { bufferView: 1, componentType: 5125, count: idx.length, type: 'SCALAR' },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: posBytes.length },
      { buffer: 0, byteOffset: posBytes.length, byteLength: idxBytes.length },
    ],
    buffers: [{ byteLength: bin.length }],
  };
  return serializeGLB({ json, bin });
}

function boxGlb(triangles) {
  const pos = new Float32Array(triangles);
  const bin = Buffer.from(pos.buffer.slice(0));
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < pos.length; i += 3)
    for (let c = 0; c < 3; c++) {
      if (pos[i + c] < min[c]) min[c] = pos[i + c];
      if (pos[i + c] > max[c]) max[c] = pos[i + c];
    }
  const json = {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
    accessors: [{ bufferView: 0, componentType: 5126, count: pos.length / 3, type: 'VEC3', min, max }],
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
const fmt = (v) => (typeof v === 'number' ? v.toFixed(4) : String(v));
function run(name, fn) {
  console.log(`\n-- ${name}`);
  const started = Date.now();
  fn();
  console.log(`   ${Date.now() - started} ms`);
}

const IDENTITY = { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] };
const trs = (position, scale = 1) => ({ position, rotation: [0, 0, 0, 1], scale: [scale, scale, scale] });
const zRot = (deg) => [0, 0, Math.sin((deg * Math.PI) / 360), Math.cos((deg * Math.PI) / 360)];

run('drape: a blanket over a bed bends at the edges and rests in the middle', () => {
  // Bed 0.4 wide, blanket 0.6 — 0.1 of overhang on each side must hang down while the
  // supported middle stays at bed height.
  const bed = new Float64Array(box(0, 0, 0, 0.4, 0.3, 0.3));
  const blanket = sheet(0.6, 0.45, 24, 18);
  const placed = trs([0, 0.22, 0]);
  const { map, report } = buildDrape({
    anchorTriangles: bed,
    placedTriangles: sheetTriangles(blanket),
    anchor: IDENTITY,
    placed,
    options: {},
  });
  check('drape solved', map !== null, report.flags.join(','));
  if (!map) return;

  // Material-point pairs: where each original vertex ended up, in world.
  const pairs = [];
  for (let i = 0; i < blanket.positions.length; i += 3) {
    const o = [blanket.positions[i], blanket.positions[i + 1], blanket.positions[i + 2]];
    const p = map(o[0], o[1], o[2]);
    pairs.push({ o, p: [p[0] + placed.position[0], p[1] + placed.position[1], p[2] + placed.position[2]] });
  }
  const bedTop = 0.15;
  const mean = (list, pick) => list.reduce((s, e) => s + pick(e), 0) / list.length;
  const middleY = mean(pairs.filter(({ o }) => Math.abs(o[0]) < 0.08 && Math.abs(o[2]) < 0.06), (e) => e.p[1]);
  const overhangY = mean(pairs.filter(({ o }) => Math.abs(o[0]) > 0.27), (e) => e.p[1]);
  const lowest = Math.min(...pairs.map(({ p }) => p[1]));
  check('middle rests on the bed', Math.abs(middleY - bedTop) < 0.035, fmt(middleY));
  check('overhang hangs well below the top', overhangY < bedTop - 0.05, fmt(overhangY));
  check('nothing fell through the ground', lowest > -0.16, fmt(lowest));
  check('no flags', report.flags.length === 0, report.flags.join(','));
  console.log(`   nodes ${report.nodes}, constraints ${report.constraints}, steps ${report.steps}`);
});

run('drape: a towel over a thin wall hangs down both sides', () => {
  const wall = new Float64Array(box(0, 0, 0, 0.06, 0.5, 0.4));
  const towel = sheet(0.4, 0.3, 20, 14);
  const placed = trs([0, 0.28, 0]);
  const { map, report } = buildDrape({
    anchorTriangles: wall,
    placedTriangles: sheetTriangles(towel),
    anchor: IDENTITY,
    placed,
  });
  check('drape solved', map !== null, report.flags.join(','));
  if (!map) return;

  // Material points: the towel halves must end below the top and wrapped close to the
  // thin wall — draped down its sides, not stretched out sideways.
  const pairs = [];
  for (let i = 0; i < towel.positions.length; i += 3) {
    const o = [towel.positions[i], towel.positions[i + 1], towel.positions[i + 2]];
    const p = map(o[0], o[1], o[2]);
    pairs.push({ o, p: [p[0], p[1] + placed.position[1], p[2]] });
  }
  const wallTop = 0.25;
  const mean = (list, pick) => list.reduce((s, e) => s + pick(e), 0) / list.length;
  const left = pairs.filter(({ o }) => o[0] < -0.15);
  const right = pairs.filter(({ o }) => o[0] > 0.15);
  check('left half hangs', mean(left, (e) => e.p[1]) < wallTop - 0.08, fmt(mean(left, (e) => e.p[1])));
  check('right half hangs', mean(right, (e) => e.p[1]) < wallTop - 0.08, fmt(mean(right, (e) => e.p[1])));
  check('hangs close to the wall', mean(left, (e) => Math.abs(e.p[0])) < 0.12 && mean(right, (e) => Math.abs(e.p[0])) < 0.12,
    `${fmt(mean(left, (e) => Math.abs(e.p[0])))}, ${fmt(mean(right, (e) => Math.abs(e.p[0])))}`);
  check('no flags', report.flags.length === 0, report.flags.join(','));
});

run('drape: a closed shell collapses instead of staying an eggshell', () => {
  // The hat-box case: a closed surface with two-sided links is rigid by geometry, so a
  // forced drape barely moved it. Tension-biased links must let it slump like an empty
  // bag — the deformed shell has to lose most of its height onto the slab below.
  const slab = new Float64Array(box(0, 0, 0, 1, 0.1, 1));
  const shell = new Float64Array(box(0, 0, 0, 0.25, 0.25, 0.25));
  const placed = trs([0, 0.2, 0]);
  const { map, report } = buildDrape({
    anchorTriangles: slab,
    placedTriangles: shell,
    anchor: IDENTITY,
    placed,
  });
  check('drape solved', map !== null, report.flags.join(','));
  if (!map) return;

  let minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < shell.length; i += 3) {
    const p = map(shell[i], shell[i + 1], shell[i + 2]);
    const y = p[1] + placed.position[1];
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const height = maxY - minY;
  check('shell collapsed to well under half its height', height < 0.125, fmt(height));
  check('came to rest on the slab', minY > 0.02 && minY < 0.09, fmt(minY));
  check('no flags', report.flags.length === 0, report.flags.join(','));
});

run('drape: a dome slumps despite its own hoop tension', () => {
  // The bowler-hat case, distilled: flattening a dome puts its hoop links in tension —
  // the arch principle — so it only collapses by pleating inward, which needs shear and
  // compression freedom. The cube-shell test cannot catch this; flat faces buckle free.
  const slab = new Float64Array(box(0, 0, 0, 1, 0.1, 1));
  const shell = [];
  const stacks = 12, slices = 32, r = 0.16;
  const at = (i, j) => {
    const phi = (i / stacks) * (Math.PI / 2);
    const theta = (j / slices) * 2 * Math.PI;
    return [r * Math.sin(phi) * Math.cos(theta), r * Math.cos(phi), r * Math.sin(phi) * Math.sin(theta)];
  };
  for (let i = 0; i < stacks; i++)
    for (let j = 0; j < slices; j++) {
      const a = at(i, j), b = at(i + 1, j), c = at(i + 1, j + 1), d = at(i, j + 1);
      if (i > 0) shell.push(...a, ...b, ...d);
      shell.push(...b, ...c, ...d);
    }
  const dome = new Float64Array(shell);
  const placed = trs([0, 0.08, 0]);
  const { map, report } = buildDrape({
    anchorTriangles: slab,
    placedTriangles: dome,
    anchor: IDENTITY,
    placed,
  });
  check('drape solved', map !== null, report.flags.join(','));
  if (!map) return;

  let minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < dome.length; i += 3) {
    const p = map(dome[i], dome[i + 1], dome[i + 2]);
    const y = p[1] + placed.position[1];
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const height = maxY - minY;
  check('dome lost at least half its height', height < 0.08, fmt(height));
  check('no flags', report.flags.length === 0, report.flags.join(','));
});

run('drape: cloth that slides off is restored to its pre-slide drape and flagged', () => {
  // The boulder case: soft binoculars on a dome slide away. Instead of refusing the
  // whole drape (rigid fallback, no ghost, no physics), the sim must keep the last
  // state that had stayed put — a drape that conformed where the phrase said.
  const steep = { position: [0, 0, 0], rotation: zRot(50), scale: [1, 1, 1] };
  const cloth = sheet(0.3, 0.25, 14, 12);
  const placed = trs([-0.2, 0.72, 0]);
  const { map, report } = buildDrape({
    anchorTriangles: new Float64Array(box(0, 0, 0, 1, 1, 1)),
    placedTriangles: sheetTriangles(cloth),
    anchor: steep,
    placed,
  });
  check('drape solved with a restore, not a failure', map !== null && !report.flags.includes('drape_failed'),
    report.flags.join(','));
  if (!map) return;
  check('flagged the refused slide', report.flags.includes('settled_away'), report.flags.join(','));

  let dx = 0, dz = 0, count = 0;
  for (let i = 0; i < cloth.positions.length; i += 3) {
    const p = map(cloth.positions[i], cloth.positions[i + 1], cloth.positions[i + 2]);
    dx += p[0] - cloth.positions[i];
    dz += p[2] - cloth.positions[i + 2];
    count++;
  }
  const wander = Math.hypot(dx / count, dz / count);
  check('kept the cloth where the phrase put it', wander < 0.16, fmt(wander));
});

run('drape: the full file path writes a deformed GLB under the same TRS', () => {
  fs.rmSync(TMP, { recursive: true, force: true });
  fs.mkdirSync(TMP, { recursive: true });
  const blanket = sheet(0.6, 0.45, 24, 18);
  fs.writeFileSync(path.join(TMP, 'bed.glb'), boxGlb(box(0, 0, 0, 0.4, 0.3, 0.3)));
  fs.writeFileSync(path.join(TMP, 'blanket.glb'), sheetGlb(blanket));

  const placedTrs = trs([0, 0.22, 0]);
  const { sizes, combined_size, report } = drapeDir(TMP, TMP, [
    { role: 'anchor', stem: 'bed', trs: IDENTITY },
    { role: 'placed', stem: 'blanket', trs: placedTrs },
  ]);
  check('reported a drape', report.contact === 'drape' && !report.flags.includes('drape_failed'), report.flags.join(','));

  const posed = parseGLB(fs.readFileSync(path.join(TMP, 'blanket.posed.glb')));
  const carried = extractTransform(posed);
  check('placement TRS preserved', carried && Math.abs(carried.position[1] - 0.22) < 1e-6,
    carried ? carried.position.join(',') : 'none');

  const world = sceneTriangles(posed);
  let minY = Infinity, maxY = -Infinity;
  for (let i = 1; i < world.length; i += 3) {
    if (world[i] < minY) minY = world[i];
    if (world[i] > maxY) maxY = world[i];
  }
  check('deformed geometry spans a real drape', maxY - minY > 0.08, fmt(maxY - minY));
  check('sizes measured off the drape', sizes.placed[1] > 0.08, fmt(sizes.placed[1]));
  check('combined box encloses both', combined_size[1] >= 0.3 - 1e-6, fmt(combined_size[1]));

  fs.rmSync(TMP, { recursive: true, force: true });
});

console.log(failures ? `\n${failures} check(s) FAILED` : '\nall checks passed');
process.exit(failures ? 1 : 0);
