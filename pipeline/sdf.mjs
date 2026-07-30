// The anchor's shape as two query structures the placement physics reads: a signed distance
// field for driving motion, and a triangle grid for exact final gaps.
//
// The SDF reuses the voxelizer's surface rasterization and outside-in flood fill, so the
// same meshes that survive stage 4 survive here — no watertightness assumed, and sealed
// interior air counts as solid, which is exactly right for contact (nothing may seat into a
// cavity that cannot be reached). Distances come from a Felzenszwalb distance transform of
// that occupancy in both directions, so the field is signed: positive outside, negative
// inside, zero within half a cell of the surface. Sampling is trilinear between cell
// centres, which keeps gradients smooth enough to act as contact normals.
//
// A cell-centre field cannot say where the surface is more precisely than half a cell, so
// anything that must *end* in exact contact finishes against the triangles themselves: the
// grid buckets them by cell and answers closest-point queries in the narrow band where the
// solver needs them.

import { bounds, rasterize, floodOutside } from './voxelize.mjs';

const PAD = 3;
const INF = 1e20;

/**
 * One pass of Felzenszwalb's lower-envelope distance transform over a strided line,
 * in place: `f` holds squared distances on entry (0 at sites, INF elsewhere) and the
 * squared distance to the nearest site on exit. `v`, `z`, `line` are scratch of length
 * ≥ n (+1 for z) shared across calls to keep the volume pass allocation-free.
 */
function edt1d(f, offset, stride, n, v, z, line) {
  for (let i = 0; i < n; i++) line[i] = f[offset + i * stride];

  let k = 0;
  v[0] = 0;
  z[0] = -INF;
  z[1] = INF;
  for (let q = 1; q < n; q++) {
    if (line[q] >= INF) continue;
    let s;
    while (true) {
      const p = v[k];
      s = (line[q] + q * q - (line[p] + p * p)) / (2 * q - 2 * p);
      if (s > z[k]) break;
      if (--k < 0) { k = 0; break; }
    }
    if (line[v[k]] >= INF) v[k] = q; // envelope so far is all-INF, replace rather than append
    else {
      k++;
      v[k] = q;
    }
    z[k] = s;
    z[k + 1] = INF;
  }

  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    const p = v[k];
    f[offset + q * stride] = (q - p) * (q - p) + line[p];
  }
}

/** Squared distance, in cells, from every cell to the nearest cell where `site` is true. */
function distanceTransform(occupancy, dims, site) {
  const [dx, dy, dz] = dims;
  const f = new Float32Array(dx * dy * dz);
  for (let i = 0; i < f.length; i++) f[i] = site(occupancy[i]) ? 0 : INF;

  const n = Math.max(dx, dy, dz);
  const v = new Int32Array(n);
  const z = new Float64Array(n + 1);
  const line = new Float64Array(n);

  // Layout is (y * dz + z) * dx + x, matching the voxelizer.
  for (let y = 0; y < dy; y++)
    for (let zz = 0; zz < dz; zz++) edt1d(f, (y * dz + zz) * dx, 1, dx, v, z, line);
  for (let y = 0; y < dy; y++)
    for (let x = 0; x < dx; x++) edt1d(f, y * dz * dx + x, dx, dz, v, z, line);
  for (let zz = 0; zz < dz; zz++)
    for (let x = 0; x < dx; x++) edt1d(f, zz * dx + x, dz * dx, dy, v, z, line);
  return f;
}

/**
 * Signed distance field of a triangle soup, `resolution` cells along its longest axis plus
 * `PAD` of clear air on every side. `phi` holds world-unit distances at cell centres:
 * positive in air, negative in the solid (interior included, courtesy of the flood fill).
 */
export function buildSDF(triangles, resolution) {
  const { min, max } = bounds(triangles);
  const size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  const h = Math.max(...size) / resolution;
  if (!(h > 0)) throw new Error('degenerate mesh bounds');

  const dims = size.map((s) => Math.max(1, Math.ceil(s / h)) + 2 * PAD);
  const gridMin = min.map((m, c) => m - ((dims[c] * h - size[c]) / 2));

  const occupancy = rasterize(triangles, dims, h, gridMin);
  floodOutside(occupancy, dims);

  const dOut = distanceTransform(occupancy, dims, (cell) => cell === 1);
  const dIn = distanceTransform(occupancy, dims, (cell) => cell === 0);

  const phi = new Float32Array(occupancy.length);
  for (let i = 0; i < phi.length; i++) {
    phi[i] = (occupancy[i] ? -Math.sqrt(dIn[i]) : Math.sqrt(dOut[i])) * h;
  }
  return { dims, h, gridMin, phi, min, max };
}

/**
 * Trilinear sampler over the field. Outside the grid the true distance is unknown but
 * bounded below: any point of the solid is at least as far as the grid box, and at most
 * `|p − clamp(p)|` closer than the clamped sample says (the field is 1-Lipschitz). The
 * larger of the two keeps queries safe for the solver's conservative stepping.
 */
export function sdfSampler({ dims, h, gridMin, phi }) {
  const [dx, dy, dz] = dims;
  const maxU = [dx - 1.001, dy - 1.001, dz - 1.001];

  const at = (px, py, pz) => {
    let ux = (px - gridMin[0]) / h - 0.5;
    let uy = (py - gridMin[1]) / h - 0.5;
    let uz = (pz - gridMin[2]) / h - 0.5;

    const cx = ux < 0 ? 0 : ux > maxU[0] ? maxU[0] : ux;
    const cy = uy < 0 ? 0 : uy > maxU[1] ? maxU[1] : uy;
    const cz = uz < 0 ? 0 : uz > maxU[2] ? maxU[2] : uz;
    const outside = Math.hypot((ux - cx) * h, (uy - cy) * h, (uz - cz) * h);

    const ix = Math.floor(cx), iy = Math.floor(cy), iz = Math.floor(cz);
    const fx = cx - ix, fy = cy - iy, fz = cz - iz;
    const base = (iy * dz + iz) * dx + ix;
    const yStride = dz * dx;

    const c000 = phi[base], c100 = phi[base + 1];
    const c010 = phi[base + yStride], c110 = phi[base + yStride + 1];
    const c001 = phi[base + dx], c101 = phi[base + dx + 1];
    const c011 = phi[base + yStride + dx], c111 = phi[base + yStride + dx + 1];

    const value =
      (c000 * (1 - fx) + c100 * fx) * (1 - fy) * (1 - fz) +
      (c001 * (1 - fx) + c101 * fx) * (1 - fy) * fz +
      (c010 * (1 - fx) + c110 * fx) * fy * (1 - fz) +
      (c011 * (1 - fx) + c111 * fx) * fy * fz;

    return outside > 0 ? Math.max(outside, value - outside) : value;
  };

  const step = h * 0.5;
  const grad = (px, py, pz, out) => {
    out[0] = at(px + step, py, pz) - at(px - step, py, pz);
    out[1] = at(px, py + step, pz) - at(px, py - step, pz);
    out[2] = at(px, py, pz + step) - at(px, py, pz - step);
    const len = Math.hypot(out[0], out[1], out[2]);
    if (len > 1e-12) {
      out[0] /= len;
      out[1] /= len;
      out[2] /= len;
    } else {
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
    }
    return out;
  };

  return { at, grad, h };
}

/** Closest point on triangle abc to p, after Ericson's Real-Time Collision Detection. */
function closestPointOnTriangle(px, py, pz, t, i, out) {
  const ax = t[i], ay = t[i + 1], az = t[i + 2];
  const bx = t[i + 3], by = t[i + 4], bz = t[i + 5];
  const cx = t[i + 6], cy = t[i + 7], cz = t[i + 8];

  const abx = bx - ax, aby = by - ay, abz = bz - az;
  const acx = cx - ax, acy = cy - ay, acz = cz - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;

  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  if (d1 <= 0 && d2 <= 0) { out[0] = ax; out[1] = ay; out[2] = az; return; }

  const bpx = px - bx, bpy = py - by, bpz = pz - bz;
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) { out[0] = bx; out[1] = by; out[2] = bz; return; }

  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    out[0] = ax + abx * v; out[1] = ay + aby * v; out[2] = az + abz * v;
    return;
  }

  const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
  const d5 = abx * cpx + aby * cpy + abz * cpz;
  const d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) { out[0] = cx; out[1] = cy; out[2] = cz; return; }

  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    out[0] = ax + acx * w; out[1] = ay + acy * w; out[2] = az + acz * w;
    return;
  }

  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / (d4 - d3 + (d5 - d6));
    out[0] = bx + (cx - bx) * w; out[1] = by + (cy - by) * w; out[2] = bz + (cz - bz) * w;
    return;
  }

  const denom = 1 / (va + vb + vc);
  const v = vb * denom;
  const w = vc * denom;
  out[0] = ax + abx * v + acx * w;
  out[1] = ay + aby * v + acy * w;
  out[2] = az + abz * v + acz * w;
}

/**
 * Triangles bucketed into a uniform grid for narrow-band closest-point queries — the SDF
 * says roughly where the surface is, this says exactly. Cells are deliberately coarser than
 * the SDF's so the lists stay short; a query walks outward shell by shell and stops as soon
 * as no unvisited cell could beat the best hit.
 */
export function buildTriangleGrid(triangles, cellSize) {
  const { min, max } = bounds(triangles);
  const dims = [0, 1, 2].map((c) => Math.max(1, Math.ceil((max[c] - min[c]) / cellSize)));
  const [dx, dy, dz] = dims;
  const cellOf = (v, c) => Math.min(dims[c] - 1, Math.max(0, Math.floor((v - min[c]) / cellSize)));

  const count = triangles.length / 9;
  const counts = new Int32Array(dx * dy * dz + 1);
  const span = new Int32Array(count * 6);

  for (let t = 0; t < count; t++) {
    const i = t * 9;
    for (let c = 0; c < 3; c++) {
      const lo = Math.min(triangles[i + c], triangles[i + 3 + c], triangles[i + 6 + c]);
      const hi = Math.max(triangles[i + c], triangles[i + 3 + c], triangles[i + 6 + c]);
      span[t * 6 + c] = cellOf(lo, c);
      span[t * 6 + 3 + c] = cellOf(hi, c);
    }
    for (let y = span[t * 6 + 1]; y <= span[t * 6 + 4]; y++)
      for (let zz = span[t * 6 + 2]; zz <= span[t * 6 + 5]; zz++)
        for (let x = span[t * 6]; x <= span[t * 6 + 3]; x++) counts[(y * dz + zz) * dx + x + 1]++;
  }
  for (let i = 1; i < counts.length; i++) counts[i] += counts[i - 1];

  const items = new Int32Array(counts[counts.length - 1]);
  const cursor = counts.slice(0, -1);
  for (let t = 0; t < count; t++) {
    for (let y = span[t * 6 + 1]; y <= span[t * 6 + 4]; y++)
      for (let zz = span[t * 6 + 2]; zz <= span[t * 6 + 5]; zz++)
        for (let x = span[t * 6]; x <= span[t * 6 + 3]; x++) items[cursor[(y * dz + zz) * dx + x]++] = t;
  }

  return { triangles, min, cellSize, dims, offsets: counts, items };
}

/**
 * Distance from `p` to the closest triangle within `maxDist`, or Infinity. `outPoint`,
 * when given, receives the witness point on the surface; `outIndex[0]` the winning
 * triangle's index — its geometry is the honest orientation of what was hit, where a
 * field gradient near corners is not.
 */
export function closestSurfaceDistance(grid, px, py, pz, maxDist, outPoint, outIndex) {
  const { triangles, min, cellSize, dims, offsets, items } = grid;
  const [dx, dy, dz] = dims;
  const cx = Math.min(dx - 1, Math.max(0, Math.floor((px - min[0]) / cellSize)));
  const cy = Math.min(dy - 1, Math.max(0, Math.floor((py - min[1]) / cellSize)));
  const cz = Math.min(dz - 1, Math.max(0, Math.floor((pz - min[2]) / cellSize)));

  let best = Infinity;
  const q = [0, 0, 0];
  const maxShell = Math.ceil(maxDist / cellSize) + 1;

  for (let r = 0; r <= maxShell; r++) {
    // Every cell of shell r is at least (r−1)·cell away, so once that exceeds the best
    // hit (or the caller's cap) the answer cannot improve.
    if ((r - 1) * cellSize > Math.min(best, maxDist)) break;

    for (let y = Math.max(0, cy - r); y <= Math.min(dy - 1, cy + r); y++) {
      for (let zz = Math.max(0, cz - r); zz <= Math.min(dz - 1, cz + r); zz++) {
        for (let x = Math.max(0, cx - r); x <= Math.min(dx - 1, cx + r); x++) {
          if (Math.max(Math.abs(x - cx), Math.abs(y - cy), Math.abs(zz - cz)) !== r) continue;
          const cell = (y * dz + zz) * dx + x;
          for (let k = offsets[cell]; k < offsets[cell + 1]; k++) {
            closestPointOnTriangle(px, py, pz, triangles, items[k] * 9, q);
            const d = Math.hypot(q[0] - px, q[1] - py, q[2] - pz);
            if (d < best) {
              best = d;
              if (outPoint) { outPoint[0] = q[0]; outPoint[1] = q[1]; outPoint[2] = q[2]; }
              if (outIndex) outIndex[0] = items[k];
            }
          }
        }
      }
    }
  }
  return best > maxDist ? Infinity : best;
}
