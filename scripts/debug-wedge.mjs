// Scratch: trace the wedge seal.

import { refinePlacement } from '../pipeline/physics.mjs';

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

const rails = new Float64Array([
  ...box(0, -0.05, 0, 1, 0.1, 1),
  ...box(-0.06, 0.05, 0, 0.02, 0.1, 1),
  ...box(0.06, 0.05, 0, 0.02, 0.1, 1),
]);

const { placed, report } = refinePlacement({
  anchorTriangles: rails,
  placedTriangles: new Float64Array(box(0, 0, 0, 0.1, 0.1, 0.1)),
  anchor: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
  placed: { position: [0, 0.03, 0], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
  intent: { contact: 'rest', direction: [0, 0, 0], embed_fraction: 0 },
});
console.log(`final y=${placed.position[1].toFixed(5)} report=${JSON.stringify(report)}`);
