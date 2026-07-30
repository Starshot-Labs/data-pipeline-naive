// Soft-body drape: the placed object as cloth. Where the rigid solver moves one body,
// this one deforms it — a blanket over a bed, a jacket across a headrest — and the trick
// that keeps it general is that the soft body is not the render mesh but its voxel
// lattice. Trellis meshes are hollow shells with arbitrary triangulation; simulated
// directly they collapse like a punctured airbed. The lattice does not care: surface
// cells of the object's own voxelization become corner nodes joined by distance
// constraints — cell edges resist stretch, face and body diagonals resist shear, and
// second-neighbour ties resist bending — and a thin object naturally yields the one-cell
// sheet that cloth wants, while a thicker one gets a coherent shell. Nodes fall under
// gravity against the anchor's distance field and the ground plane with friction, and
// the render mesh follows by trilinear embedding, textures and detail untouched.
//
// The result leaves this module as a `map` over file-frame positions (fed to
// `deformGLB`), so the deformation is expressed in the mesh's own local space under the
// unchanged placement TRS — every pipeline invariant about poses living in files holds.
//
// Same contract as the rigid pass: refine a roughly-correct answer, never rescue a wrong
// one. A drape that blows up, wanders off, or never comes to rest reports `drape_failed`
// and the caller bakes the model's rigid answer instead.

import { buildSDF, sdfSampler, buildTriangleGrid, closestSurfaceDistance } from './sdf.mjs';
import { rasterize, bounds } from './voxelize.mjs';

const DEFAULTS = { drapeRes: 24, sdfRes: 128 };

const GRAVITY = 6;
const SUBSTEP = 1 / 240;
const MAX_SUBSTEPS = 3000;
const ITERATIONS = 2;
const DAMPING = 0.996;
const V_MAX = 1.5;
const FRICTION = 1.0;
// Fabric fights stretch, shears without much argument, folds under compression almost
// freely, and holds only a whisper of bending stiffness. The asymmetry is what lets a
// closed shell collapse at all — links that resist compression as firmly as stretch
// make an eggshell (Cauchy rigidity), which is how a draped hat box failed to slump —
// and low shear is what lets a dome escape its own hoop tension by pleating inward,
// which is how a draped bowler hat failed to slump. Bend ties stay bilateral: they are
// the small stiffness that keeps folds rounded instead of creased to zero.
const STRETCH_K = 1.0;
const SHEAR_K = 0.5;
const BEND_K = 0.15;
const COMPRESS = 0.02;
const SLEEP_V = 0.02;
const SLEEP_STEPS = 60;
const WARMUP = 30;
const CONTACT_BIAS = 0.002;

const qRotate = (q, x, y, z, out) => {
  const tx = 2 * (q[1] * z - q[2] * y);
  const ty = 2 * (q[2] * x - q[0] * z);
  const tz = 2 * (q[0] * y - q[1] * x);
  out[0] = x + q[3] * tx + q[1] * tz - q[2] * ty;
  out[1] = y + q[3] * ty + q[2] * tx - q[0] * tz;
  out[2] = z + q[3] * tz + q[0] * ty - q[1] * tx;
  return out;
};

/**
 * Drape the placed mesh over the anchor. Returns `{ map, report }`: `map` transforms a
 * position in the placed file's own frame to its deformed position in that same frame,
 * or is null when the drape refused and the rigid answer should stand.
 */
export function buildDrape({ anchorTriangles, placedTriangles, anchor, placed, options = {} }) {
  const report = { contact: 'drape', flags: [] };

  // Anchor-normalized world, exactly like the rigid pass.
  const aWorld = applyTRS(anchorTriangles, anchor);
  const aBounds = bounds(aWorld);
  const aSize = Math.max(
    aBounds.max[0] - aBounds.min[0],
    aBounds.max[1] - aBounds.min[1],
    aBounds.max[2] - aBounds.min[2],
  );
  if (!(aSize > 0)) throw new Error('degenerate anchor bounds');
  const norm = 1 / aSize;
  for (let i = 0; i < aWorld.length; i++) aWorld[i] *= norm;

  const field = sdfSampler(buildSDF(aWorld, Number(options.sdfRes ?? DEFAULTS.sdfRes)));
  const grid = buildTriangleGrid(aWorld, 2 * field.h);
  const ground = aBounds.min[1] * norm;

  // The lattice lives in the placed file's local frame: surface-rasterized occupancy
  // (no fill, no erosion — a sheet must never vanish), dilated once so every render
  // vertex lands in an occupied cell.
  const lattice = buildLattice(placedTriangles, Number(options.drapeRes ?? DEFAULTS.drapeRes));
  const { cell, gridMin, nodeCoords } = lattice;
  const nodeCount = nodeCoords.length / 3;
  report.nodes = nodeCount;

  // Node rest state in normalized world. Uniform scale is a pipeline invariant.
  const scale = placed.scale[0];
  const q = placed.rotation;
  const t = placed.position;
  const pos = new Float64Array(nodeCount * 3);
  const scratch = [0, 0, 0];
  for (let n = 0; n < nodeCount; n++) {
    const lx = gridMin[0] + (nodeCoords[n * 3] + 0.5) * cell;
    const ly = gridMin[1] + (nodeCoords[n * 3 + 1] + 0.5) * cell;
    const lz = gridMin[2] + (nodeCoords[n * 3 + 2] + 0.5) * cell;
    qRotate(q, lx * scale, ly * scale, lz * scale, scratch);
    pos[n * 3] = (scratch[0] + t[0]) * norm;
    pos[n * 3 + 1] = (scratch[1] + t[1]) * norm;
    pos[n * 3 + 2] = (scratch[2] + t[2]) * norm;
  }
  const start = Float64Array.from(pos);

  const constraints = buildConstraints(lattice, pos);
  report.constraints = constraints.length / 5;

  // Rigid pre-drop to first contact, so the simulation drapes instead of falling.
  const margin = 0.35 * cell * scale * norm;
  const clearance = (n) =>
    Math.min(field.at(pos[n * 3], pos[n * 3 + 1], pos[n * 3 + 2]) - margin, pos[n * 3 + 1] - ground);
  for (let i = 0; i < 64; i++) {
    let d = Infinity;
    for (let n = 0; n < nodeCount; n++) d = Math.min(d, clearance(n));
    if (d <= 1e-4) break;
    for (let n = 0; n < nodeCount; n++) pos[n * 3 + 1] -= d;
  }

  const outcome = simulate(pos, constraints, field, ground, margin, lattice, norm, report);
  if (!outcome.ok) {
    report.flags.push('drape_failed');
    return { map: null, report };
  }
  report.steps = outcome.steps;
  report.flags.push(...outcome.flags);

  // A cell-quantized collider leaves the cloth hovering by its margin plus the field's
  // fattening; one vertical seal against the actual triangles closes it. The gap is
  // measured on the deformed render vertices themselves — the lattice nodes sit half a
  // cell off the surface by construction.
  const skin = makeSkin(lattice, pos);
  let exact = Infinity;
  for (let i = 0; i < placedTriangles.length; i += 3) {
    const p = skin(placedTriangles[i], placedTriangles[i + 1], placedTriangles[i + 2]);
    const plane = p[1] - ground;
    if (plane < exact) exact = plane;
    if (field.at(p[0], p[1], p[2]) < 6 * field.h) {
      const d = closestSurfaceDistance(grid, p[0], p[1], p[2], 6 * field.h);
      if (d < exact) exact = d;
    }
  }
  if (Number.isFinite(exact) && exact > 0) {
    const drop = Math.min(exact + CONTACT_BIAS, 4 * field.h + margin);
    for (let n = 0; n < nodeCount; n++) pos[n * 3 + 1] -= drop;
  }

  let moved = 0;
  let lowest = Infinity;
  for (let n = 0; n < nodeCount; n++) {
    moved += Math.hypot(
      pos[n * 3] - start[n * 3],
      pos[n * 3 + 1] - start[n * 3 + 1],
      pos[n * 3 + 2] - start[n * 3 + 2],
    );
    if (pos[n * 3 + 1] < lowest) lowest = pos[n * 3 + 1];
  }
  report.moved = Number(((moved / nodeCount) * aSize).toFixed(6));
  report.gap_after = Number((measureGap(pos, nodeCount, field, margin) * aSize).toFixed(6));

  // World-deformed → back through the inverse TRS, so the file keeps its own pose node.
  const qInv = [-q[0], -q[1], -q[2], q[3]];
  const map = (x, y, z) => {
    const world = skin(x, y, z);
    const wx = world[0] / norm - t[0];
    const wy = world[1] / norm - t[1];
    const wz = world[2] / norm - t[2];
    qRotate(qInv, wx, wy, wz, scratch);
    return [scratch[0] / scale, scratch[1] / scale, scratch[2] / scale];
  };
  return { map, report };
}

const applyTRS = (triangles, { position, rotation, scale }) => {
  const out = new Float64Array(triangles.length);
  const v = [0, 0, 0];
  for (let i = 0; i < triangles.length; i += 3) {
    qRotate(rotation, triangles[i] * scale[0], triangles[i + 1] * scale[1], triangles[i + 2] * scale[2], v);
    out[i] = v[0] + position[0];
    out[i + 1] = v[1] + position[1];
    out[i + 2] = v[2] + position[2];
  }
  return out;
};

/**
 * Surface occupancy of the placed mesh at `resolution` cells along its longest side,
 * with a node at the centre of every occupied cell. Rasterization only — no flood fill,
 * no fill-ratio erosion, no dilation: a sheet one cell thick must become a single layer
 * of nodes, which is exactly the classic mass-spring cloth. Corner nodes were tried and
 * rejected — they give even the thinnest sheet two layers, and a double-layer truss with
 * inextensible edges is a plate that will not bend. Thicker objects still come out as a
 * volumetric centre-node lattice; the structure is the same, only deeper.
 */
function buildLattice(triangles, resolution) {
  const { min, max } = bounds(triangles);
  const size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  const cell = Math.max(...size) / resolution;
  if (!(cell > 0)) throw new Error('degenerate placed bounds');
  const dims = size.map((s) => Math.max(1, Math.ceil(s / cell)) + 2);
  const gridMin = min.map((m, c) => m - ((dims[c] * cell - size[c]) / 2));

  const occupied = rasterize(triangles, dims, cell, gridMin);
  const [dx, dy, dz] = dims;

  const nodeOf = new Int32Array(occupied.length).fill(-1);
  const coords = [];
  for (let y = 0; y < dy; y++)
    for (let z = 0; z < dz; z++)
      for (let x = 0; x < dx; x++) {
        const cellIndex = (y * dz + z) * dx + x;
        if (!occupied[cellIndex]) continue;
        nodeOf[cellIndex] = coords.length / 3;
        coords.push(x, y, z);
      }

  return { dims, cell, gridMin, occupied, nodeOf, nodeCoords: Int32Array.from(coords) };
}

// Neighbour offsets by role: axis links resist stretch, diagonals resist shear (and keep
// a staircase-rasterized sheet connected), two-apart ties resist bending. Only positive
// directions — every pair is visited from its lower end once.
const STRETCH_LINKS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
const SHEAR_LINKS = [
  [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1],
  [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
];
const BEND_LINKS = [[2, 0, 0], [0, 2, 0], [0, 0, 2]];

/** Distance constraints as a flat [a, b, rest, tensionK, compressK] list. */
function buildConstraints(lattice, pos) {
  const { dims, nodeOf, nodeCoords } = lattice;
  const [dx, dy, dz] = dims;
  const node = (x, y, z) =>
    x >= 0 && y >= 0 && z >= 0 && x < dx && y < dy && z < dz ? nodeOf[(y * dz + z) * dx + x] : -1;

  const out = [];
  const link = (a, b, tensionK, compressK) => {
    if (b < 0) return;
    const rest = Math.hypot(
      pos[a * 3] - pos[b * 3],
      pos[a * 3 + 1] - pos[b * 3 + 1],
      pos[a * 3 + 2] - pos[b * 3 + 2],
    );
    out.push(a, b, rest, tensionK, compressK);
  };

  for (let a = 0; a < nodeCoords.length / 3; a++) {
    const x = nodeCoords[a * 3], y = nodeCoords[a * 3 + 1], z = nodeCoords[a * 3 + 2];
    for (const [ox, oy, oz] of STRETCH_LINKS) link(a, node(x + ox, y + oy, z + oz), STRETCH_K, STRETCH_K * COMPRESS);
    for (const [ox, oy, oz] of SHEAR_LINKS) link(a, node(x + ox, y + oy, z + oz), SHEAR_K, SHEAR_K * COMPRESS);
    for (const [ox, oy, oz] of BEND_LINKS) link(a, node(x + ox, y + oy, z + oz), BEND_K, BEND_K);
  }
  return Float64Array.from(out);
}

/**
 * The drape itself: PBD particles, distance constraints, field + plane contacts. The
 * same instruction-first policy as the rigid settle: cloth may conform where the phrase
 * put it, not slide away. A rolling snapshot keeps the last state that had wandered less
 * than half the cap, and a breach restores it — a drape that stayed put, flagged — while
 * a numeric blow-up still fails outright and hands the caller the rigid fallback.
 */
function simulate(pos, constraints, field, ground, margin, lattice, norm, report) {
  const count = pos.length / 3;
  const vel = new Float64Array(pos.length);
  const prev = new Float64Array(pos.length);
  const snapshot = Float64Array.from(pos);
  const n = [0, 0, 0];

  let cx0 = 0, cz0 = 0;
  for (let i = 0; i < count; i++) {
    cx0 += pos[i * 3];
    cz0 += pos[i * 3 + 2];
  }
  cx0 /= count;
  cz0 /= count;
  const wanderCap = 0.15;

  let calm = 0;
  let steps = 0;
  for (; steps < MAX_SUBSTEPS; steps++) {
    prev.set(pos);
    for (let i = 0; i < count; i++) {
      vel[i * 3 + 1] -= GRAVITY * SUBSTEP;
      for (let c = 0; c < 3; c++) {
        vel[i * 3 + c] *= DAMPING;
        const v = vel[i * 3 + c];
        vel[i * 3 + c] = v > V_MAX ? V_MAX : v < -V_MAX ? -V_MAX : v;
        pos[i * 3 + c] += vel[i * 3 + c] * SUBSTEP;
      }
    }

    for (let iter = 0; iter < ITERATIONS; iter++) {
      for (let k = 0; k < constraints.length; k += 5) {
        const a = constraints[k] * 3;
        const b = constraints[k + 1] * 3;
        const rest = constraints[k + 2];
        const dx = pos[b] - pos[a];
        const dy = pos[b + 1] - pos[a + 1];
        const dz = pos[b + 2] - pos[a + 2];
        const d = Math.hypot(dx, dy, dz);
        if (d < 1e-12) continue;
        const err = d - rest;
        const s = ((err > 0 ? constraints[k + 3] : constraints[k + 4]) * 0.5 * err) / d;
        pos[a] += dx * s;
        pos[a + 1] += dy * s;
        pos[a + 2] += dz * s;
        pos[b] -= dx * s;
        pos[b + 1] -= dy * s;
        pos[b + 2] -= dz * s;
      }
    }

    for (let i = 0; i < count; i++) {
      const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
      let depth = margin - field.at(x, y, z);
      if (depth > 0) {
        field.grad(x, y, z, n);
        pos[i * 3] += n[0] * depth;
        pos[i * 3 + 1] += n[1] * depth;
        pos[i * 3 + 2] += n[2] * depth;
        applyFriction(pos, prev, i, n, depth);
      }
      const plane = pos[i * 3 + 1] - ground;
      if (plane < 0) {
        pos[i * 3 + 1] -= plane;
        applyFriction(pos, prev, i, [0, 1, 0], -plane);
      }
    }

    let maxSpeed = 0;
    let cx = 0, cz = 0;
    for (let i = 0; i < count; i++) {
      for (let c = 0; c < 3; c++) vel[i * 3 + c] = (pos[i * 3 + c] - prev[i * 3 + c]) / SUBSTEP;
      const speed = Math.hypot(vel[i * 3], vel[i * 3 + 1], vel[i * 3 + 2]);
      if (speed > maxSpeed) maxSpeed = speed;
      cx += pos[i * 3];
      cz += pos[i * 3 + 2];
    }

    if (!Number.isFinite(maxSpeed)) return { ok: false };
    const wander = Math.hypot(cx / count - cx0, cz / count - cz0);
    if (wander > wanderCap) {
      pos.set(snapshot);
      return { ok: true, steps, flags: ['settled_away'] };
    }
    if (wander < wanderCap / 2 && steps % 20 === 0) snapshot.set(pos);

    if (steps > WARMUP && maxSpeed < SLEEP_V * 2) {
      if (++calm >= SLEEP_STEPS) break;
    } else calm = 0;
  }
  return { ok: true, steps, flags: [] };
}

/** Cancel this substep's tangential slip at a contact, Coulomb-capped by its depth. */
function applyFriction(pos, prev, i, n, depth) {
  let sx = pos[i * 3] - prev[i * 3];
  let sy = pos[i * 3 + 1] - prev[i * 3 + 1];
  let sz = pos[i * 3 + 2] - prev[i * 3 + 2];
  const along = sx * n[0] + sy * n[1] + sz * n[2];
  sx -= n[0] * along;
  sy -= n[1] * along;
  sz -= n[2] * along;
  const slip = Math.hypot(sx, sy, sz);
  if (slip < 1e-12) return;
  const k = Math.min(1, (FRICTION * depth) / slip);
  pos[i * 3] -= sx * k;
  pos[i * 3 + 1] -= sy * k;
  pos[i * 3 + 2] -= sz * k;
}

const measureGap = (pos, count, field, margin) => {
  let min = Infinity;
  for (let n = 0; n < count; n++) {
    const phi = field.at(pos[n * 3], pos[n * 3 + 1], pos[n * 3 + 2]) - margin;
    if (phi < min) min = phi;
  }
  return min;
};

/**
 * Trilinear embedding of a file-frame position in the deformed lattice, staggered over
 * cell-centre nodes: the eight nearest centres carry it, with weights renormalized over
 * the ones that exist — at a sheet's edge half of them are open space. A position whose
 * neighbourhood is entirely empty (boundary rounding) rides the nearest occupied node.
 */
function makeSkin(lattice, pos) {
  const { dims, cell, gridMin, occupied, nodeOf } = lattice;
  const [dx, dy, dz] = dims;
  const clampCell = (v, d) => (v < 0 ? 0 : v > d - 1 ? d - 1 : v);

  const nearestNode = (x, y, z) => {
    for (let r = 0; r <= 3; r++) {
      for (let oy = -r; oy <= r; oy++)
        for (let oz = -r; oz <= r; oz++)
          for (let ox = -r; ox <= r; ox++) {
            const nx = clampCell(x + ox, dx), ny = clampCell(y + oy, dy), nz = clampCell(z + oz, dz);
            const node = nodeOf[(ny * dz + nz) * dx + nx];
            if (node >= 0) return node;
          }
    }
    return -1;
  };

  return (fx, fy, fz) => {
    const ux = (fx - gridMin[0]) / cell - 0.5;
    const uy = (fy - gridMin[1]) / cell - 0.5;
    const uz = (fz - gridMin[2]) / cell - 0.5;
    const bx = Math.floor(ux), by = Math.floor(uy), bz = Math.floor(uz);
    const tx = ux - bx, ty = uy - by, tz = uz - bz;

    let x = 0, y = 0, z = 0, total = 0;
    for (let oy = 0; oy <= 1; oy++)
      for (let oz = 0; oz <= 1; oz++)
        for (let ox = 0; ox <= 1; ox++) {
          const cxi = bx + ox, cyi = by + oy, czi = bz + oz;
          if (cxi < 0 || cyi < 0 || czi < 0 || cxi >= dx || cyi >= dy || czi >= dz) continue;
          const node = nodeOf[(cyi * dz + czi) * dx + cxi];
          if (node < 0) continue;
          const w = (ox ? tx : 1 - tx) * (oy ? ty : 1 - ty) * (oz ? tz : 1 - tz);
          if (w <= 0) continue;
          total += w;
          x += w * pos[node * 3];
          y += w * pos[node * 3 + 1];
          z += w * pos[node * 3 + 2];
        }
    if (total > 1e-9) return [x / total, y / total, z / total];

    const node = nearestNode(clampCell(Math.round(ux), dx), clampCell(Math.round(uy), dy), clampCell(Math.round(uz), dz));
    return node >= 0 ? [pos[node * 3], pos[node * 3 + 1], pos[node * 3 + 2]] : [fx, fy, fz];
  };
}
