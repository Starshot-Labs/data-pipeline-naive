// Numerical checks for the placement physics, on synthetic meshes — the corpus lives on
// the volume, so correctness has to be provable from geometry built right here. Every mode
// gets a scene with a known right answer and the solver's output is measured against it.
//
//   node scripts/test-physics.mjs

import { refinePlacement } from '../pipeline/physics.mjs';
import { buildSDF, sdfSampler } from '../pipeline/sdf.mjs';

/* --------------------------------- mesh builders ---------------------------------- */

/** Axis-aligned box as 12 triangles, centred at (cx, cy, cz), full sizes (sx, sy, sz). */
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

/** UV sphere triangles, centred at (cx, cy, cz). */
function sphere(cx, cy, cz, r, stacks = 24, slices = 48) {
  const at = (i, j) => {
    const phi = (i / stacks) * Math.PI;
    const theta = (j / slices) * 2 * Math.PI;
    return [
      cx + r * Math.sin(phi) * Math.cos(theta),
      cy + r * Math.cos(phi),
      cz + r * Math.sin(phi) * Math.sin(theta),
    ];
  };
  const out = [];
  for (let i = 0; i < stacks; i++) {
    for (let j = 0; j < slices; j++) {
      const a = at(i, j), b = at(i + 1, j), c = at(i + 1, j + 1), d = at(i, j + 1);
      if (i > 0) out.push(...a, ...b, ...d);
      if (i < stacks - 1) out.push(...b, ...c, ...d);
    }
  }
  return out;
}

/** A five-sided open container: outer 1×1×1, walls and floor 0.1 thick, open at +Y. */
function openBox() {
  return [
    ...box(0, -0.45, 0, 1, 0.1, 1),
    ...box(-0.45, 0.05, 0, 0.1, 0.9, 1),
    ...box(0.45, 0.05, 0, 0.1, 0.9, 1),
    ...box(0, 0.05, -0.45, 0.8, 0.9, 0.1),
    ...box(0, 0.05, 0.45, 0.8, 0.9, 0.1),
  ];
}

const soup = (arr) => new Float64Array(arr);
const IDENTITY = { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] };
const trs = (position, rotation = [0, 0, 0, 1], scale = [1, 1, 1]) => ({ position, rotation, scale });
const zRot = (deg) => [0, 0, Math.sin((deg * Math.PI) / 360), Math.cos((deg * Math.PI) / 360)];

/** World bounds of a mesh under a TRS, for asserting where the solver actually put it. */
function worldBounds(triangles, { position, rotation, scale }) {
  const [qx, qy, qz, qw] = rotation;
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < triangles.length; i += 3) {
    const x = triangles[i] * scale[0], y = triangles[i + 1] * scale[1], z = triangles[i + 2] * scale[2];
    const tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
    const p = [
      x + qw * tx + qy * tz - qz * ty + position[0],
      y + qw * ty + qz * tx - qx * tz + position[1],
      z + qw * tz + qx * ty - qy * tx + position[2],
    ];
    for (let c = 0; c < 3; c++) {
      if (p[c] < min[c]) min[c] = p[c];
      if (p[c] > max[c]) max[c] = p[c];
    }
  }
  return { min, max };
}

/* ------------------------------------ harness ------------------------------------- */

let failures = 0;
function check(label, ok, detail = '') {
  if (!ok) failures++;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${label}${detail ? `  (${detail})` : ''}`);
}
const near = (value, want, tol) => Math.abs(value - want) <= tol;
const fmt = (v) => (typeof v === 'number' ? v.toFixed(4) : String(v));

function run(name, fn) {
  console.log(`\n-- ${name}`);
  const started = Date.now();
  fn();
  console.log(`   ${Date.now() - started} ms`);
}

/* ------------------------------------- tests -------------------------------------- */

run('sdf: field of a unit cube reads distances', () => {
  // In the padded band the field is cell-accurate (surface apparent up to one fattened
  // cell out); beyond the pad it degrades to a safe lower bound — never an overestimate.
  const field = sdfSampler(buildSDF(soup(box(0, 0, 0, 1, 1, 1)), 64));
  const h = field.h;
  const nearFace = field.at(0, 0.53, 0);
  check('just outside the top face is small and positive', nearFace > 0 && nearFace < 2.5 * h, fmt(nearFace));
  const atFace = field.at(0, 0.5, 0);
  check('the face itself reads within the occupancy quantum', atFace > -2.5 * h && atFace < 0.5 * h, fmt(atFace));
  check('deep inside is negative', field.at(0, 0, 0) < -0.3, fmt(field.at(0, 0, 0)));
  const far = field.at(0, 0.75, 0);
  check('beyond the pad is a lower bound, never more', far > 0.15 && far <= 0.25 + h, fmt(far));
  check('gradient direction is sane', field.at(0, 0.6, 0) > field.at(0, 0.53, 0));
});

const cube = soup(box(0, 0, 0, 1, 1, 1));
const smallCube = soup(box(0, 0, 0, 0.1, 0.1, 0.1));

run('rest: a floating cube drops onto the anchor top', () => {
  const { placed, report } = refinePlacement({
    anchorTriangles: cube,
    placedTriangles: smallCube,
    anchor: IDENTITY,
    placed: trs([0, 0.58, 0]),
    intent: { contact: 'rest', direction: [0, 0, 0], embed_fraction: 0 },
  });
  const { min } = worldBounds(smallCube, placed);
  check('bottom meets the top face', near(min[1], 0.5 - 0.002, 0.008), fmt(min[1]));
  check('did not slide sideways', near(placed.position[0], 0, 0.012) && near(placed.position[2], 0, 0.012),
    `${fmt(placed.position[0])}, ${fmt(placed.position[2])}`);
  check('stayed upright', report.rotated_degrees < 4, `${report.rotated_degrees}°`);
  check('no flags', report.flags.length === 0, report.flags.join(','));
});

run('rest: a cube sunk into the anchor pops back out', () => {
  const { placed, report } = refinePlacement({
    anchorTriangles: cube,
    placedTriangles: smallCube,
    anchor: IDENTITY,
    placed: trs([0, 0.52, 0]),
    intent: { contact: 'rest', direction: [0, 0, 0], embed_fraction: 0 },
  });
  const { min } = worldBounds(smallCube, placed);
  check('bottom ends on the top face', near(min[1], 0.5 - 0.002, 0.01), fmt(min[1]));
  check('reported the initial penetration', report.gap_before < -0.02, fmt(report.gap_before));
});

run('rest: a ball dropped inside an open container lands on its floor', () => {
  const container = soup(openBox());
  const ball = soup(sphere(0, 0, 0, 0.1));
  const { placed, report } = refinePlacement({
    anchorTriangles: container,
    placedTriangles: ball,
    anchor: IDENTITY,
    placed: trs([0.15, 0.2, 0]),
    intent: { contact: 'rest', direction: [0, 0, 0], embed_fraction: 0 },
  });
  check('rests on the cavity floor', near(placed.position[1], -0.3, 0.015), fmt(placed.position[1]));
  check('still inside the cavity', Math.abs(placed.position[0]) < 0.3 && Math.abs(placed.position[2]) < 0.3,
    `${fmt(placed.position[0])}, ${fmt(placed.position[2])}`);
  check('no flags', report.flags.length === 0, report.flags.join(','));
});

run('lean: a tilted plank meets both the ground and the wall face', () => {
  const wall = soup(box(0, 0, 0, 0.2, 1, 1));
  const plank = soup(box(0, 0, 0, 0.05, 0.7, 0.2));
  const { placed, report } = refinePlacement({
    anchorTriangles: wall,
    placedTriangles: plank,
    anchor: IDENTITY,
    // Base out at x=0.32, tipped 20° toward the wall, floating a hair off the ground.
    placed: trs([0.32, -0.13, 0], zRot(20)),
    intent: { contact: 'lean', direction: [-1, 0, 0], embed_fraction: 0 },
  });
  const { min } = worldBounds(plank, placed);
  check('base is on the ground', near(min[1], -0.5, 0.012), fmt(min[1]));
  check('top touches the wall face at x=0.1', near(min[0], 0.1 - 0.002, 0.015), fmt(min[0]));
  check('not flagged incomplete', !report.flags.includes('lean_incomplete'), report.flags.join(','));
});

run('attach: a cube mounts flush onto the side face, gravity ignored', () => {
  const { placed, report } = refinePlacement({
    anchorTriangles: cube,
    placedTriangles: smallCube,
    anchor: IDENTITY,
    placed: trs([0.59, 0.2, 0]),
    intent: { contact: 'attach', direction: [-1, 0, 0], embed_fraction: 0 },
  });
  const { min } = worldBounds(smallCube, placed);
  check('face meets the wall at x=0.5', near(min[0], 0.5 - 0.002, 0.008), fmt(min[0]));
  check('did not fall', near(placed.position[1], 0.2, 0.005), fmt(placed.position[1]));
  check('moved about the gap', near(report.moved, 0.04, 0.01), fmt(report.moved));
});

run('attach: a hanging cube closes the gap upward', () => {
  const slabWithPost = soup([...box(0, 0.4, 0, 1, 0.2, 1), ...box(0.4, -0.25, 0.4, 0.1, 1.1, 0.1)]);
  const { placed } = refinePlacement({
    anchorTriangles: slabWithPost,
    placedTriangles: smallCube,
    anchor: IDENTITY,
    placed: trs([0, 0.22, 0]),
    intent: { contact: 'attach', direction: [0, 1, 0], embed_fraction: 0 },
  });
  const { max } = worldBounds(smallCube, placed);
  check('top meets the slab underside at y=0.3', near(max[1], 0.3 + 0.002, 0.008), fmt(max[1]));
});

run('embed: a rod buries a quarter of its length', () => {
  const rod = soup(box(0, 0, 0, 0.04, 0.4, 0.04));
  const { placed, report } = refinePlacement({
    anchorTriangles: cube,
    placedTriangles: rod,
    anchor: IDENTITY,
    placed: trs([0, 0.72, 0]),
    intent: { contact: 'embed', direction: [0, -1, 0], embed_fraction: 0.25 },
  });
  const { min } = worldBounds(rod, placed);
  check('tip ends 0.1 below the surface', near(min[1], 0.4, 0.02), fmt(min[1]));
  check('no flags', report.flags.length === 0, report.flags.join(','));
});

run('none: the answer is left untouched', () => {
  const before = trs([0, 0.9, 0]);
  const { placed, report } = refinePlacement({
    anchorTriangles: cube,
    placedTriangles: smallCube,
    anchor: IDENTITY,
    placed: before,
    intent: { contact: 'none', direction: [0, 0, 0], embed_fraction: 0 },
  });
  check('transform unchanged', placed === before);
  check('nothing reported moved', report.moved === 0);
});

run('guards: buried starts and unreachable contacts revert with a flag', () => {
  const buried = refinePlacement({
    anchorTriangles: cube,
    placedTriangles: smallCube,
    anchor: IDENTITY,
    placed: trs([0, 0, 0]),
    intent: { contact: 'rest', direction: [0, 0, 0], embed_fraction: 0 },
  });
  check('deep start is refused', buried.report.flags.includes('deep_penetration'), buried.report.flags.join(','));
  check('deep start keeps the answer', buried.placed.position[1] === 0);

  const away = refinePlacement({
    anchorTriangles: cube,
    placedTriangles: smallCube,
    anchor: IDENTITY,
    placed: trs([5, 0, 0]),
    intent: { contact: 'attach', direction: [1, 0, 0], embed_fraction: 0 },
  });
  check('cast away from A finds nothing', away.report.flags.includes('no_contact'), away.report.flags.join(','));
  check('failed cast keeps the answer', away.placed.position[0] === 5);
});

run('rest: a plank dropped level onto a mild slope rotates to lie flush', () => {
  // The dragon-skull complaint: dropping is not enough, the body must conform. A 20°
  // face holds under friction, so the settle should tip the plank onto it — rotation
  // about the first-touching edge, which the wander budget must not mistake for drift.
  const slope = trs([0, 0, 0], zRot(20));
  const plank = soup(box(0, 0, 0, 0.3, 0.02, 0.1));
  const { report } = refinePlacement({
    anchorTriangles: cube,
    placedTriangles: plank,
    anchor: slope,
    placed: trs([-0.17, 0.62, 0]),
    intent: { contact: 'rest', direction: [0, 0, 0], embed_fraction: 0 },
  });
  check('rotated to the face', report.rotated_degrees > 10 && report.rotated_degrees < 30, `${report.rotated_degrees}°`);
  check('was not mistaken for drift', !report.flags.includes('settled_away'), report.flags.join(','));
  check('ended in contact', report.gap_after < 0.008, fmt(report.gap_after));
});

run('rest: a slope that would slide the body away is refused in favour of the drop', () => {
  // A cube dropped onto a 50°-tilted face: honest physics slides it downhill and away
  // from where the phrase put it. The instruction wins — the solver keeps the dropped
  // pose (vertical motion only) and flags that the settle wanted to leave.
  const slope = trs([0, 0, 0], zRot(50));
  const { placed, report } = refinePlacement({
    anchorTriangles: cube,
    placedTriangles: smallCube,
    anchor: slope,
    placed: trs([0.1, 0.75, 0]),
    intent: { contact: 'rest', direction: [0, 0, 0], embed_fraction: 0 },
  });
  check('kept the lateral position', near(placed.position[0], 0.1, 0.035) && near(placed.position[2], 0, 0.035),
    `${fmt(placed.position[0])}, ${fmt(placed.position[2])}`);
  check('still dropped into contact', report.gap_after < 0.005, fmt(report.gap_after));
  check('flagged the refused slide', report.flags.includes('settled_away'), report.flags.join(','));
});

run('rest: a body tucked into a snug recess stays tucked', () => {
  // A 0.1 cube dropped to the bottom of a deep 0.102-wide well: the hairline clearance
  // is far under the field's one-cell fattening, so every wall reads as contact at any
  // height inside the well. Reaching clear air would mean lifting the cube all the way
  // out — the escape-pod-hatch failure — so the pose must stand, flagged nested.
  const half = 0.051;
  const plate = soup([
    ...box(0, -0.2, 0, 1, 0.1, 1),
    ...box(0, 0, (half + 0.5) / 2, 1, 0.3, 0.5 - half),
    ...box(0, 0, -(half + 0.5) / 2, 1, 0.3, 0.5 - half),
    ...box((half + 0.5) / 2, 0, 0, 0.5 - half, 0.3, 2 * half),
    ...box(-(half + 0.5) / 2, 0, 0, 0.5 - half, 0.3, 2 * half),
  ]);
  const { placed, report } = refinePlacement({
    anchorTriangles: plate,
    placedTriangles: smallCube,
    anchor: IDENTITY,
    placed: trs([0, -0.1, 0]),
    intent: { contact: 'rest', direction: [0, 0, 0], embed_fraction: 0 },
  });
  check('stayed at the bottom of the well', near(placed.position[1], -0.1, 0.012), fmt(placed.position[1]));
  check('did not shift sideways', near(placed.position[0], 0, 0.005) && near(placed.position[2], 0, 0.005),
    `${fmt(placed.position[0])}, ${fmt(placed.position[2])}`);
  check('flagged as nested', report.flags.includes('nested'), report.flags.join(','));
});

run('rest: a body wedged over a ridge surfaces by the amount that helps', () => {
  // A 0.1 cube sunk 0.02 into the floor of a snug groove — 0.004 of side clearance,
  // well under the field's fattening, so the rails read as contact at every height and
  // clear air is unreachable. But lifting a couple of cells relieves the real overlap
  // underneath. The dragon-skull case: it must surface onto the true floor instead of
  // being left sunken as "nested".
  const rails = soup([
    ...box(0, -0.05, 0, 1, 0.1, 1),
    ...box(-0.064, 0.05, 0, 0.02, 0.1, 1),
    ...box(0.064, 0.05, 0, 0.02, 0.1, 1),
  ]);
  const { placed, report } = refinePlacement({
    anchorTriangles: rails,
    placedTriangles: smallCube,
    anchor: IDENTITY,
    placed: trs([0, 0.03, 0]),
    intent: { contact: 'rest', direction: [0, 0, 0], embed_fraction: 0 },
  });
  const { min } = worldBounds(smallCube, placed);
  check('bottom surfaced onto the groove floor', near(min[1], -0.002, 0.009), fmt(min[1]));
  check('did not shift sideways', near(placed.position[0], 0, 0.005) && near(placed.position[2], 0, 0.005),
    `${fmt(placed.position[0])}, ${fmt(placed.position[2])}`);
  check('flagged as nested', report.flags.includes('nested'), report.flags.join(','));
});

run('rest: a deep drop into a container lands on A and is not flagged', () => {
  // Same container, ball released high above the floor: the fall is long but ends on the
  // anchor, which is a container doing its job — no dropped_far, no drifted.
  const container = soup(openBox());
  const ball = soup(sphere(0, 0, 0, 0.1));
  const { placed, report } = refinePlacement({
    anchorTriangles: container,
    placedTriangles: ball,
    anchor: IDENTITY,
    placed: trs([-0.1, 0.35, 0.05]),
    intent: { contact: 'rest', direction: [0, 0, 0], embed_fraction: 0 },
  });
  check('rests on the cavity floor', near(placed.position[1], -0.3, 0.015), fmt(placed.position[1]));
  check('no flags on a legitimate long drop', report.flags.length === 0, report.flags.join(','));
});

run('perf: a dense anchor stays well under budget', () => {
  const dense = soup(sphere(0, 0, 0, 0.5, 100, 200));
  const ball = soup(sphere(0, 0, 0, 0.08, 16, 32));
  const started = Date.now();
  const { placed, report } = refinePlacement({
    anchorTriangles: dense,
    placedTriangles: ball,
    anchor: IDENTITY,
    placed: trs([0, 0.72, 0]),
    intent: { contact: 'attach', direction: [0, -1, 0], embed_fraction: 0 },
  });
  const ms = Date.now() - started;
  console.log(`   ${dense.length / 9} anchor triangles, ${ms} ms`);
  const { min } = worldBounds(ball, placed);
  check('single scene under 2.5 s', ms < 2500, `${ms} ms`);
  check('ball bottom meets the pole at y=0.5', near(min[1], 0.5 - 0.002, 0.01), fmt(min[1]));
  check('no flags', report.flags.length === 0, report.flags.join(','));
});

console.log(failures ? `\n${failures} check(s) FAILED` : '\nall checks passed');
process.exit(failures ? 1 : 0);
