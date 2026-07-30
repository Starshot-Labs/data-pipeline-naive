// Step 5.5: mesh-accurate placement refinement. The placement model reasons over voxel
// blocks a sixteenth of the anchor wide, so its answer is right in intent and off in
// contact — the placed object floats a sliver away from the surface it should touch, or
// sinks a sliver into it. This pass moves the placed object, and only the placed object,
// the last few percent using the true triangles of both meshes.
//
// What "right" means depends on the phrase, so the placement model classifies its own
// answer (`contact` in the schema) and this solver acts accordingly:
//
//   rest    gravity holds B against A or the ground: drop it, let it tip and roll on the
//           real surface until it settles (a rigid body on the true meshes).
//   lean    gravity presses B against the ground and a face of A: settle with gravity
//           tilted toward that face, so both contacts engage and stay.
//   attach  B is held touching A with no help from gravity (mounted, hanging, stuck on):
//           slide B along the approach direction into exact contact, nothing else.
//   embed   B intentionally penetrates A (dart in a board): slide along the approach
//           direction until the asked-for fraction of B is buried.
//   none    no contact intended (hovering, floating): leave the answer alone.
//
// The machinery is two primitives over one representation. A becomes a signed distance
// field plus a triangle grid (sdf.mjs); B becomes a few thousand area-weighted surface
// points. `seat` sphere-traces those points along a direction until the field says touch,
// and `settle` runs a positional rigid-body integrator (small-substep PBD with friction)
// against the field. Both end with a nudge against A's actual triangles, because a
// cell-sized field cannot say "touching" more precisely than half a cell.
//
// The solver is deliberately conservative: it refines a roughly-correct answer and refuses
// to rescue a wrong one. Motion beyond its caps, contact it cannot find, or a start buried
// deep inside A all revert to the model's answer with a flag in the report, so a batch run
// can filter rather than inherit the damage. Scale is never touched — the body is rigid.

import fs from 'node:fs';
import path from 'node:path';
import { parseGLB, sceneTriangles } from './glb.mjs';
import { buildSDF, sdfSampler, buildTriangleGrid, closestSurfaceDistance } from './sdf.mjs';

// Everything below is in anchor-normalized units: the anchor's longest side is 1, so a
// constant reads as a fraction of the anchor. Times are seconds of simulated fall.
const DEFAULTS = { sdfRes: 128, samples: 2000 };

const CONTACT_BIAS = 0.002;   // finish this far inside the surface, so contact reads as contact
const SEAT_TOL = 5e-4;        // how close the cast converges on its target gap
const SEAT_MAX_TRAVEL = 0.9;  // a cast that finds nothing within this never finds anything
const BACKOUT_CELLS = 6;      // a back-out longer than this is un-nesting, not fixing voxel error
const SNAP_MAX = 0.03;        // close a residual gap to the other surface only when it is this small
const SNAP_MIN = 0.004;       // ...and leave gaps below this alone: they already read as contact
const LEAN_TILT = 0.35;       // sideways fraction of gravity that presses a leaning body home

const GRAVITY = 4;
const SUBSTEP = 1 / 240;
const MAX_SUBSTEPS = 1200;
// High friction on purpose: this poses bodies, and a placement should stick where the
// phrase put it on any slope a real object would plausibly rest on (μ=1 holds 45°).
const FRICTION = 1.0;
const V_MAX = 1.5;
const W_MAX = 25;
// Sleep must not mistake the start of a tip for rest — that verdict lives in the tight
// angular threshold, because tipping is rotation. The linear threshold sits just above
// the contact noise floor (a resting body micro-bounces at gravity-per-substep, about
// 0.017) or nothing would ever sleep. A body still creeping slower than CREEP_V when
// the substep budget runs out is rest enough.
const SLEEP_V = 0.02;
const SLEEP_W = 0.05;
const SLEEP_STEPS = 60;
const CREEP_V = 0.1;
// Soft depenetration: contacts may correct the pose only this much per substep, so an
// embedded start resolves over many substeps instead of one violent one — the velocity
// derived from positions stays physical wherever the body began.
const MAX_CORRECTION = 0.003;
const MAX_ANG_CORRECTION = 0.03;
// A settle may conform — tip, rock, roll a little — but not topple: beyond this the
// pose no longer resembles the answer it was refining.
const ROTATION_CAP = Math.PI / 3;

const unit = (v) => {
  const len = Math.hypot(v[0], v[1], v[2]);
  return len > 1e-9 ? [v[0] / len, v[1] / len, v[2] / len] : null;
};

/* ---------------------------------- quaternions ---------------------------------- */

const qIdentity = () => [0, 0, 0, 1];

function qRotate(q, x, y, z, out) {
  const tx = 2 * (q[1] * z - q[2] * y);
  const ty = 2 * (q[2] * x - q[0] * z);
  const tz = 2 * (q[0] * y - q[1] * x);
  out[0] = x + q[3] * tx + q[1] * tz - q[2] * ty;
  out[1] = y + q[3] * ty + q[2] * tx - q[0] * tz;
  out[2] = z + q[3] * tz + q[0] * ty - q[1] * tx;
  return out;
}

const qRotateInv = (q, x, y, z, out) => {
  const inv = [-q[0], -q[1], -q[2], q[3]];
  return qRotate(inv, x, y, z, out);
};

function qMul(a, b, out) {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  out[0] = aw * bx + ax * bw + ay * bz - az * by;
  out[1] = aw * by + ay * bw + az * bx - ax * bz;
  out[2] = aw * bz + az * bw + ax * by - ay * bx;
  out[3] = aw * bw - ax * bx - ay * by - az * bz;
  return out;
}

function qNormalize(q) {
  const len = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  q[0] /= len; q[1] /= len; q[2] /= len; q[3] /= len;
  return q;
}

/** q ← q rotated by the small world-space vector w·dt (first-order integration). */
function qApplyDelta(q, wx, wy, wz, dt) {
  const half = 0.5 * dt;
  const dx = wx * half, dy = wy * half, dz = wz * half;
  const [x, y, z, w] = q;
  q[0] = x + (dx * w + dy * z - dz * y);
  q[1] = y + (dy * w + dz * x - dx * z);
  q[2] = z + (dz * w + dx * y - dy * x);
  q[3] = w + (-dx * x - dy * y - dz * z);
  return qNormalize(q);
}

const qAngleDeg = (q) => (2 * Math.acos(Math.min(1, Math.abs(q[3])))) * (180 / Math.PI);

/* ------------------------------------ the body ------------------------------------ */

const mulberry32 = (seed) => () => {
  seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

/**
 * The placed object as the solver sees it: `local` holds sample points relative to the
 * centroid (the body frame at its starting orientation), `x`/`q` are the pose delta the
 * solver accumulates, and `iInv` is the inverse inertia of the sample shell about the
 * centroid — mass 1, so contacts and gravity need no other constants.
 */
function makeBody(worldTriangles, count) {
  const triCount = worldTriangles.length / 9;
  const cumulative = new Float64Array(triCount);
  let total = 0;
  for (let t = 0; t < triCount; t++) {
    const i = t * 9;
    const ux = worldTriangles[i + 3] - worldTriangles[i], uy = worldTriangles[i + 4] - worldTriangles[i + 1], uz = worldTriangles[i + 5] - worldTriangles[i + 2];
    const vx = worldTriangles[i + 6] - worldTriangles[i], vy = worldTriangles[i + 7] - worldTriangles[i + 1], vz = worldTriangles[i + 8] - worldTriangles[i + 2];
    const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
    total += Math.hypot(cx, cy, cz) / 2;
    cumulative[t] = total;
  }
  if (!(total > 0)) throw new Error('placed mesh has no surface area');

  const random = mulberry32(0x9e3779b9 ^ triCount ^ count);
  const points = new Float64Array(count * 3);
  for (let s = 0; s < count; s++) {
    const pick = random() * total;
    let lo = 0, hi = triCount - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cumulative[mid] < pick) lo = mid + 1;
      else hi = mid;
    }
    const i = lo * 9;
    const su = Math.sqrt(random());
    const u = 1 - su;
    const v = random() * su;
    for (let c = 0; c < 3; c++) {
      points[s * 3 + c] =
        worldTriangles[i + c] * u + worldTriangles[i + 3 + c] * v + worldTriangles[i + 6 + c] * (1 - u - v);
    }
  }

  const centroid = [0, 0, 0];
  for (let s = 0; s < count; s++) for (let c = 0; c < 3; c++) centroid[c] += points[s * 3 + c];
  for (let c = 0; c < 3; c++) centroid[c] /= count;

  const local = new Float64Array(count * 3);
  let radius = 0;
  let ixx = 0, iyy = 0, izz = 0, ixy = 0, ixz = 0, iyz = 0;
  for (let s = 0; s < count; s++) {
    const x = points[s * 3] - centroid[0];
    const y = points[s * 3 + 1] - centroid[1];
    const z = points[s * 3 + 2] - centroid[2];
    local[s * 3] = x; local[s * 3 + 1] = y; local[s * 3 + 2] = z;
    radius = Math.max(radius, Math.hypot(x, y, z));
    ixx += y * y + z * z; iyy += x * x + z * z; izz += x * x + y * y;
    ixy -= x * y; ixz -= x * z; iyz -= y * z;
  }
  const m = 1 / count;
  ixx *= m; iyy *= m; izz *= m; ixy *= m; ixz *= m; iyz *= m;
  // A shell of samples can be nearly line-like; the ridge keeps the inverse bounded.
  const ridge = 0.02 * ((ixx + iyy + izz) / 3) + 1e-9;
  ixx += ridge; iyy += ridge; izz += ridge;

  const det =
    ixx * (iyy * izz - iyz * iyz) - ixy * (ixy * izz - iyz * ixz) + ixz * (ixy * iyz - iyy * ixz);
  const d = 1 / det;
  const iInv = [
    (iyy * izz - iyz * iyz) * d, (ixz * iyz - ixy * izz) * d, (ixy * iyz - ixz * iyy) * d,
    (ixz * iyz - ixy * izz) * d, (ixx * izz - ixz * ixz) * d, (ixy * ixz - ixx * iyz) * d,
    (ixy * iyz - ixz * iyy) * d, (ixy * ixz - ixx * iyz) * d, (ixx * iyy - ixy * ixy) * d,
  ];

  return { count, local, x: [...centroid], q: qIdentity(), start: [...centroid], iInv, radius };
}

/** World position of sample `s` under the body's current pose, into `out`. */
function samplePoint(body, s, out) {
  qRotate(body.q, body.local[s * 3], body.local[s * 3 + 1], body.local[s * 3 + 2], out);
  out[0] += body.x[0];
  out[1] += body.x[1];
  out[2] += body.x[2];
  return out;
}

/** iInv is body-frame, so the world torque makes a round trip through the orientation. */
function applyInvInertia(body, x, y, z, out) {
  qRotateInv(body.q, x, y, z, out);
  const [bx, by, bz] = out;
  const m = body.iInv;
  const rx = m[0] * bx + m[1] * by + m[2] * bz;
  const ry = m[3] * bx + m[4] * by + m[5] * bz;
  const rz = m[6] * bx + m[7] * by + m[8] * bz;
  return qRotate(body.q, rx, ry, rz, out);
}

/* --------------------------------- the primitives --------------------------------- */

/** min over samples of the field, at the body's pose shifted by `t·dir`. */
function minField(body, field, dir, t, ground) {
  const p = [0, 0, 0];
  let min = Infinity;
  for (let s = 0; s < body.count; s++) {
    samplePoint(body, s, p);
    const value = field.at(p[0] + dir[0] * t, p[1] + dir[1] * t, p[2] + dir[2] * t);
    if (value < min) min = value;
    if (ground !== null) {
      const pd = p[1] + dir[1] * t - ground;
      if (pd < min) min = pd;
    }
  }
  return min;
}

/**
 * Slide the body along `dir` until the deepest sample reads `target` (0 is touch, negative
 * is intentional burial). Forward motion sphere-traces — the field is 1-Lipschitz, so
 * stepping by the remaining margin cannot pass through anything it should stop at; a start
 * already past the target backs out the same way. Either way the final approach is a
 * bisection, because a cell-quantized field is only monotone to within its own noise.
 *
 * The back-out is capped separately and much shorter than the approach: it exists to fix
 * a voxel-sized error, and a body that cannot reach clear air within `maxBackout` is not
 * mis-placed but *nested* — tucked into a recess tighter than the field can resolve, where
 * every wall reads as contact. Backing that out would lift the body clean out of the
 * recess, so the cast refuses (and moves nothing) instead.
 */
function seat(body, field, dir, target, ground, maxBackout = SEAT_MAX_TRAVEL) {
  let t = 0;
  let g = minField(body, field, dir, 0, ground);
  const before = g;

  if (g > target) {
    let travelled = 0;
    for (let i = 0; i < 128 && g - target > SEAT_TOL; i++) {
      const step = Math.max(g - target, SEAT_TOL);
      t += step;
      travelled += step;
      if (travelled > SEAT_MAX_TRAVEL) return { ok: false, before };
      g = minField(body, field, dir, t, ground);
      if (g < target - SEAT_TOL) {
        let lo = t - step, hi = t;
        for (let b = 0; b < 32; b++) {
          const mid = (lo + hi) / 2;
          if (minField(body, field, dir, mid, ground) > target) lo = mid;
          else hi = mid;
        }
        t = hi;
        break;
      }
    }
  } else {
    // Already at or past the target depth: back out along −dir until clear, then bisect
    // down onto the crossing. `t = 0` is a valid deep end of the bracket by construction;
    // the last expansion probes exactly at the cap before giving up.
    let lo = -Math.min(Math.max(target - g, field.h), maxBackout);
    for (let i = 0; i < 64 && minField(body, field, dir, lo, ground) <= target; i++) {
      if (-lo >= maxBackout) return { ok: false, before };
      lo = -Math.min(-lo * 1.6, maxBackout);
    }
    let hi = 0;
    for (let b = 0; b < 32; b++) {
      const mid = (lo + hi) / 2;
      if (minField(body, field, dir, mid, ground) > target) lo = mid;
      else hi = mid;
    }
    t = (lo + hi) / 2;
  }

  body.x[0] += dir[0] * t;
  body.x[1] += dir[1] * t;
  body.x[2] += dir[2] * t;
  return { ok: true, before, moved: t };
}

/**
 * Let gravity finish the placement: a positional rigid-body integrator in the style of
 * small-substep XPBD. Each substep predicts the pose, projects every penetrating sample
 * out of the field along its gradient (with the proper generalized mass, so pushes off-
 * centre become torques), applies Coulomb friction as a positional correction against the
 * substep's slip, and re-derives velocities from what actually moved. Zero restitution
 * falls out of that for free, which is exactly right for coming to rest.
 *
 * Candidate contacts are cached: all samples are classified every time the body has moved
 * far enough that the cache could be stale, and only nearby ones are tested in between —
 * a resting body re-tests a handful of points, not thousands.
 */
function settle(body, field, gravity, ground, wanderCap) {
  const v = [0, 0, 0];
  const w = [0, 0, 0];
  const p = [0, 0, 0];
  const pPrev = [0, 0, 0];
  const n = [0, 0, 0];
  const scratch = [0, 0, 0];
  const arm = [0, 0, 0];
  const accX = [0, 0, 0];
  const accT = [0, 0, 0];

  const slack = 10 * field.h;
  let candidates = [];
  let travelledSinceRefresh = Infinity;

  // Wander is measured at the support, not the centre: a body conforming to the surface
  // tips about its lowest point, which barely moves, while a slide or a roll carries it
  // away — the same centroid displacement, opposite verdicts. The entry orientation
  // anchors the topple check.
  const q0 = [...body.q];
  let pivot = 0;
  let lowest = Infinity;
  for (let s = 0; s < body.count; s++) {
    samplePoint(body, s, p);
    if (p[1] < lowest) {
      lowest = p[1];
      pivot = s;
    }
  }
  samplePoint(body, pivot, p);
  const pivotX = p[0];
  const pivotZ = p[2];

  let calm = 0;
  let steps = 0;
  let contactsSeen = false;
  let lastContact = -Infinity;
  const flags = [];

  for (; steps < MAX_SUBSTEPS; steps++) {
    v[0] += gravity[0] * SUBSTEP;
    v[1] += gravity[1] * SUBSTEP;
    v[2] += gravity[2] * SUBSTEP;
    const speed = Math.hypot(v[0], v[1], v[2]);
    if (speed > V_MAX) for (let c = 0; c < 3; c++) v[c] *= V_MAX / speed;
    const spin = Math.hypot(w[0], w[1], w[2]);
    if (spin > W_MAX) for (let c = 0; c < 3; c++) w[c] *= W_MAX / spin;
    // Heavy angular damping is deliberate: this poses a body, it does not simulate one,
    // and a sample-faceted ball would otherwise creep-roll on flat ground for seconds.
    for (let c = 0; c < 3; c++) { v[c] *= 0.998; w[c] *= 0.99; }

    const xPrev = [...body.x];
    const qPrev = [...body.q];
    body.x[0] += v[0] * SUBSTEP;
    body.x[1] += v[1] * SUBSTEP;
    body.x[2] += v[2] * SUBSTEP;
    qApplyDelta(body.q, w[0], w[1], w[2], SUBSTEP);

    travelledSinceRefresh += speed * SUBSTEP + spin * SUBSTEP * body.radius;
    if (travelledSinceRefresh > slack - 2 * field.h) {
      candidates = [];
      for (let s = 0; s < body.count; s++) {
        samplePoint(body, s, p);
        const near = field.at(p[0], p[1], p[2]) < slack || (ground !== null && p[1] - ground < slack);
        if (near) candidates.push(s);
      }
      travelledSinceRefresh = 0;
    }

    // Contacts resolve Jacobi-style: every penetrating sample contributes its own full
    // positional correction against the same predicted pose, and the mean is applied
    // once. Sequential impulses would depend on visit order and slowly walk or rock the
    // body; the mean is order-free, and for the common case — a face flat on a surface,
    // where every contact asks for nearly the same correction — it resolves in one step.
    // Anything less coherent finishes over the following substeps.
    let contacts = 0;
    accX[0] = accX[1] = accX[2] = 0;
    accT[0] = accT[1] = accT[2] = 0;
    for (const s of candidates) {
      samplePoint(body, s, p);

      for (let which = 0; which < 2; which++) {
        let phi;
        if (which === 0) {
          phi = field.at(p[0], p[1], p[2]);
          if (phi >= 0) continue;
          field.grad(p[0], p[1], p[2], n);
        } else {
          if (ground === null) continue;
          phi = p[1] - ground;
          if (phi >= 0) continue;
          n[0] = 0; n[1] = 1; n[2] = 0;
        }
        contacts++;

        arm[0] = p[0] - body.x[0];
        arm[1] = p[1] - body.x[1];
        arm[2] = p[2] - body.x[2];
        const cx = arm[1] * n[2] - arm[2] * n[1];
        const cy = arm[2] * n[0] - arm[0] * n[2];
        const cz = arm[0] * n[1] - arm[1] * n[0];
        applyInvInertia(body, cx, cy, cz, scratch);
        const wGen = 1 + cx * scratch[0] + cy * scratch[1] + cz * scratch[2];
        const dLambda = -phi / wGen;

        accX[0] += n[0] * dLambda;
        accX[1] += n[1] * dLambda;
        accX[2] += n[2] * dLambda;
        accT[0] += cx * dLambda;
        accT[1] += cy * dLambda;
        accT[2] += cz * dLambda;

        // Static friction, positionally: cancel this substep's tangential slip at the
        // contact, up to the Coulomb cone of the normal correction it rode in with.
        qRotate(qPrev, body.local[s * 3], body.local[s * 3 + 1], body.local[s * 3 + 2], pPrev);
        let sx = p[0] - pPrev[0] - xPrev[0];
        let sy = p[1] - pPrev[1] - xPrev[1];
        let sz = p[2] - pPrev[2] - xPrev[2];
        const along = sx * n[0] + sy * n[1] + sz * n[2];
        sx -= n[0] * along; sy -= n[1] * along; sz -= n[2] * along;
        const slip = Math.hypot(sx, sy, sz);
        if (slip > 1e-9) {
          const limit = Math.min(slip, FRICTION * dLambda);
          const tx = sx / slip, ty = sy / slip, tz = sz / slip;
          const fx = arm[1] * tz - arm[2] * ty;
          const fy = arm[2] * tx - arm[0] * tz;
          const fz = arm[0] * ty - arm[1] * tx;
          applyInvInertia(body, fx, fy, fz, scratch);
          const wT = 1 + fx * scratch[0] + fy * scratch[1] + fz * scratch[2];
          const dT = limit / wT;
          accX[0] -= tx * dT;
          accX[1] -= ty * dT;
          accX[2] -= tz * dT;
          accT[0] -= fx * dT;
          accT[1] -= fy * dT;
          accT[2] -= fz * dT;
        }
      }
    }

    if (contacts > 0) {
      contactsSeen = true;
      lastContact = steps;
      let mx = accX[0] / contacts, my = accX[1] / contacts, mz = accX[2] / contacts;
      const mag = Math.hypot(mx, my, mz);
      if (mag > MAX_CORRECTION) {
        const k = MAX_CORRECTION / mag;
        mx *= k; my *= k; mz *= k;
      }
      body.x[0] += mx;
      body.x[1] += my;
      body.x[2] += mz;
      applyInvInertia(body, accT[0] / contacts, accT[1] / contacts, accT[2] / contacts, scratch);
      const ang = Math.hypot(scratch[0], scratch[1], scratch[2]);
      if (ang > MAX_ANG_CORRECTION) {
        const k = MAX_ANG_CORRECTION / ang;
        scratch[0] *= k; scratch[1] *= k; scratch[2] *= k;
      }
      qApplyDelta(body.q, scratch[0], scratch[1], scratch[2], 1);
    }

    for (let c = 0; c < 3; c++) v[c] = (body.x[c] - xPrev[c]) / SUBSTEP;
    const dq = qMul(body.q, [-qPrev[0], -qPrev[1], -qPrev[2], qPrev[3]], [0, 0, 0, 1]);
    const sign = dq[3] < 0 ? -1 : 1;
    for (let c = 0; c < 3; c++) w[c] = (2 * dq[c] * sign) / SUBSTEP;

    // Pivot travel and topple are capped but falls are not: a drop just closes the gap
    // the model left (the ground plane bounds it anyway), while a body that slides,
    // rolls away or keels over no longer sits where — or how — the phrase said it does.
    // Better to keep the answer and flag it than bake a placement that stopped matching
    // its own instruction.
    samplePoint(body, pivot, p);
    const drift = Math.hypot(p[0] - pivotX, p[2] - pivotZ);
    const dot = Math.abs(
      body.q[0] * q0[0] + body.q[1] * q0[1] + body.q[2] * q0[2] + body.q[3] * q0[3],
    );
    if (drift > wanderCap || 2 * Math.acos(Math.min(1, dot)) > ROTATION_CAP) {
      return { steps, flags: ['drifted'], slept: false };
    }

    if (steps > 10 && Math.hypot(...v) < SLEEP_V && Math.hypot(...w) < SLEEP_W) {
      if (++calm >= SLEEP_STEPS) break;
    } else calm = 0;
  }

  // Running out of substeps only matters when the body is genuinely still going — a slow
  // creep while supported is rest for our purposes, just not by the strict sleep test.
  // "Supported" means contact within the last few substeps: a body at rest micro-bounces,
  // so the final substep alone is a coin flip on the bounce phase.
  if (steps >= MAX_SUBSTEPS && !(steps - lastContact <= 4 && Math.hypot(...v) < CREEP_V)) {
    flags.push('no_rest');
  }
  if (!contactsSeen) flags.push('no_contact');
  return { steps, flags, slept: steps < MAX_SUBSTEPS };
}

/* --------------------------------- exact contact ---------------------------------- */

/**
 * The gap left along `dir`, read off the actual triangles. Two filters keep the unsigned
 * closest-point distance honest as a directional measure: the sample's field gradient
 * must oppose the motion, and the witness point itself must lie along the motion axis —
 * either ahead (a genuine gap) or behind (already touching, which pins the gap near zero
 * and stops a repeat push). Without the second filter a surface grazing the body's side,
 * like a groove wall a hair away, reads as an almost-closed gap and stalls the motion
 * far from the surface it is actually heading for. Infinity when nothing relevant is in
 * the narrow band.
 */
function directionalGap(body, field, grid, dir, ground, band = 5) {
  const p = [0, 0, 0];
  const n = [0, 0, 0];
  const w = [0, 0, 0];
  const hit = [0];
  let gap = Infinity;
  for (let s = 0; s < body.count; s++) {
    samplePoint(body, s, p);
    if (field.at(p[0], p[1], p[2]) <= band * field.h) {
      field.grad(p[0], p[1], p[2], n);
      if (n[0] * dir[0] + n[1] * dir[1] + n[2] * dir[2] <= -0.5) {
        const d = closestSurfaceDistance(grid, p[0], p[1], p[2], band * field.h, w, hit);
        if (d < gap) {
          if (d < CONTACT_BIAS * 0.5) {
            // Effectively touching: at this range the witness offers no direction and
            // the field gradient lies near corners — the touched triangle's own plane
            // is the truth. It pins the gap only when it squarely opposes the motion;
            // a wall grazing the body's side must not stop a descent.
            const t = grid.triangles;
            const i = hit[0] * 9;
            const ux = t[i + 3] - t[i], uy = t[i + 4] - t[i + 1], uz = t[i + 5] - t[i + 2];
            const vx = t[i + 6] - t[i], vy = t[i + 7] - t[i + 1], vz = t[i + 8] - t[i + 2];
            const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
            const len = Math.hypot(cx, cy, cz);
            const blocking = len > 1e-12 &&
              Math.abs(cx * dir[0] + cy * dir[1] + cz * dir[2]) >= 0.8 * len;
            if (blocking) gap = d;
          } else {
            const along = (w[0] - p[0]) * dir[0] + (w[1] - p[1]) * dir[1] + (w[2] - p[2]) * dir[2];
            if (Math.abs(along) >= 0.5 * d) gap = d;
          }
        }
      }
    }
    if (ground !== null && dir[1] < -0.5) {
      const pd = p[1] - ground;
      if (pd < gap) gap = pd;
    }
  }
  return gap;
}

/**
 * Translate along `dir` until the directional gap closes, measured on the actual
 * triangles rather than the field. This is what runs once any contact already exists:
 * the field reads an established touch as slightly negative everywhere near it, so a
 * field-driven cast in a second direction would refuse to move — while the unsigned
 * triangle distance never overshoots (the straight-line gap is a lower bound on travel
 * along any direction), so stepping by it converges from below and ends with the
 * `CONTACT_BIAS` hop that makes the baked scene read as touching rather than almost.
 * Ends without contact — nothing within `band` cells, or the travel budget spent —
 * put everything back.
 */
function pushToContact(body, field, grid, dir, ground, maxTravel, band) {
  let moved = 0;
  const translate = (t) => {
    body.x[0] += dir[0] * t;
    body.x[1] += dir[1] * t;
    body.x[2] += dir[2] * t;
    moved += t;
  };

  for (let i = 0; i < 8; i++) {
    const gap = directionalGap(body, field, grid, dir, ground, band);
    if (!Number.isFinite(gap)) break;
    if (gap <= SNAP_MIN) {
      const hop = Math.min(gap + CONTACT_BIAS, maxTravel - moved);
      if (hop > 0) translate(hop);
      return { ok: true, moved };
    }
    const step = Math.min(gap, maxTravel - moved);
    if (step <= 0) break;
    translate(step);
  }
  translate(-moved);
  return { ok: false, moved: 0 };
}

/** Smallest distance from any sample to A's triangles (unsigned, capped), for reporting. */
function measureGap(body, field, grid) {
  const p = [0, 0, 0];
  let sdfMin = Infinity;
  let exact = Infinity;
  for (let s = 0; s < body.count; s++) {
    samplePoint(body, s, p);
    const phi = field.at(p[0], p[1], p[2]);
    if (phi < sdfMin) sdfMin = phi;
    if (phi < 6 * field.h) {
      const d = closestSurfaceDistance(grid, p[0], p[1], p[2], 6 * field.h);
      if (d < exact) exact = d;
    }
  }
  return sdfMin < 0 ? sdfMin : Number.isFinite(exact) ? exact : sdfMin;
}

/** Direction from B's closest sample toward A, read straight off the field's gradient. */
function towardAnchor(body, field) {
  const p = [0, 0, 0];
  const n = [0, 0, 0];
  let best = Infinity;
  let at = null;
  for (let s = 0; s < body.count; s++) {
    samplePoint(body, s, p);
    const phi = field.at(p[0], p[1], p[2]);
    if (phi < best) {
      best = phi;
      at = [...p];
    }
  }
  if (!at) return null;
  field.grad(at[0], at[1], at[2], n);
  return unit([-n[0], -n[1], -n[2]]);
}

/**
 * A long fall that ends on the anchor is a container doing its job; one that ends on
 * the bare ground plane means the body missed A entirely — probably slid off an edge
 * the model thought it was on. Worth a flag so a batch can filter on it.
 */
function flagDroppedFar(body, field, report) {
  const drop = body.start[1] - body.x[1];
  if (drop <= Math.max(0.35, 3 * body.radius)) return;
  const p = [0, 0, 0];
  for (let s = 0; s < body.count; s++) {
    samplePoint(body, s, p);
    if (field.at(p[0], p[1], p[2]) < field.h) return;
  }
  report.flags.push('dropped_far');
}

/* ----------------------------------- the solver ----------------------------------- */

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
 * Refine the placed object's transform against the anchor's actual surface.
 *
 * Takes both meshes' triangles in their own file frames, the world TRS each was assigned,
 * and the placement model's declared intent; returns the placed TRS with the solver's
 * rigid correction folded in, plus a report of what it did. The anchor never moves.
 */
export function refinePlacement({ anchorTriangles, placedTriangles, anchor, placed, intent = {}, options = {} }) {
  const contact = intent.contact ?? 'attach';
  const report = { contact, flags: [], moved: 0, rotated_degrees: 0 };
  if (contact === 'none') return { placed, report };

  const sdfRes = Number(options.sdfRes ?? DEFAULTS.sdfRes);
  const sampleCount = Number(options.samples ?? DEFAULTS.samples);

  // Anchor-normalized frame: apply both TRS, then divide the world by the anchor's longest
  // side so every constant above is a fraction of the anchor.
  const aWorld = applyTRS(anchorTriangles, anchor);
  let aMin = [Infinity, Infinity, Infinity], aMax = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < aWorld.length; i += 3) {
    for (let c = 0; c < 3; c++) {
      if (aWorld[i + c] < aMin[c]) aMin[c] = aWorld[i + c];
      if (aWorld[i + c] > aMax[c]) aMax[c] = aWorld[i + c];
    }
  }
  const aSize = Math.max(aMax[0] - aMin[0], aMax[1] - aMin[1], aMax[2] - aMin[2]);
  if (!(aSize > 0)) throw new Error('degenerate anchor bounds');
  const norm = 1 / aSize;
  for (let i = 0; i < aWorld.length; i++) aWorld[i] *= norm;

  const bWorld = applyTRS(placedTriangles, placed);
  for (let i = 0; i < bWorld.length; i++) bWorld[i] *= norm;

  const field = sdfSampler(buildSDF(aWorld, sdfRes));
  const grid = buildTriangleGrid(aWorld, 2 * field.h);
  const body = makeBody(bWorld, sampleCount);
  const ground = aMin[1] * norm;

  const gapBefore = measureGap(body, field, grid);
  report.gap_before = gapBefore * aSize;

  // A body mostly buried in the anchor is a wrong answer, not a gap — refuse it.
  const bDiag = 2 * body.radius;
  if (gapBefore < -0.5 * bDiag) {
    report.flags.push('deep_penetration');
    report.gap_after = report.gap_before;
    return { placed, report };
  }

  const given = Array.isArray(intent.direction) ? unit(intent.direction) : null;
  const down = [0, -1, 0];

  if (contact === 'attach' || contact === 'embed') {
    const dir = given ?? towardAnchor(body, field);
    if (!dir) report.flags.push('no_direction');
    else {
      let target = 0;
      if (contact === 'embed') {
        let lo = Infinity, hi = -Infinity;
        for (let s = 0; s < body.count; s++) {
          const along = body.local[s * 3] * dir[0] + body.local[s * 3 + 1] * dir[1] + body.local[s * 3 + 2] * dir[2];
          if (along < lo) lo = along;
          if (along > hi) hi = along;
        }
        const fraction = Math.min(Math.max(Number(intent.embed_fraction) || 0, 0.02), 0.5);
        target = -Math.min(fraction * (hi - lo), 0.45 * (hi - lo));
      }
      const seated = seat(body, field, dir, target, null, BACKOUT_CELLS * field.h);
      if (!seated.ok) {
        // Starting in contact and unable to back clear means nested, not lost: the body
        // sits in a recess tighter than the field resolves, and the model's pose stands.
        report.flags.push(seated.before < 0 ? 'nested' : 'no_contact');
        report.gap_after = report.gap_before;
        return { placed, report };
      }
      if (contact === 'attach') {
        pushToContact(body, field, grid, dir, null, 4 * field.h, 5);
        // A mounted object that stops a hair above the ground was meant to meet it too.
        const p = [0, 0, 0];
        let planeGap = Infinity;
        for (let s = 0; s < body.count; s++) planeGap = Math.min(planeGap, samplePoint(body, s, p)[1] - ground);
        if (planeGap > SNAP_MIN && planeGap < SNAP_MAX) {
          pushToContact(body, field, grid, down, ground, SNAP_MAX + 2 * field.h, 8);
        }
      }
    }
  } else if (contact === 'rest') {
    // Drop first, dynamics second. The vertical seat is the correction the phrase
    // actually needs — it closes the gap the model left without moving the body an inch
    // sideways, and its back-out branch handles a penetrating start as a straight lift.
    // The settle that follows only gets to stabilize the pose locally: tip flat, rock,
    // roll to a stop. Where the true surface turns out sloped or domed, honest physics
    // would slide the body away from where the phrase put it — the instruction wins,
    // so a slide beyond the budget is undone and flagged rather than baked.
    // Vertical placement first: a floating body drops until it touches; a body the
    // field reads as overlapping lifts to the least-penetrating height within a few
    // cells. The scan serves every start with one mechanism — a voxel-sized error
    // clears completely, a body wedged against a lumpy ridge rises the few cells that
    // relieve the real overlap underneath, and one tucked into a snug recess peaks at
    // zero lift and stays (anything that still cannot reach clear air is flagged
    // nested). Either way the body ends at the field's zero or its best approximation,
    // a hair above the true surface — the kiss the settle must start from.
    const start = minField(body, field, down, 0, ground);
    if (start > 0) {
      seat(body, field, down, 0, ground);
    } else {
      const cap = BACKOUT_CELLS * field.h;
      const up = [0, 1, 0];
      let bestT = 0;
      let bestPhi = start;
      for (let i = 1; i <= 24; i++) {
        const t = (cap * i) / 24;
        const phi = minField(body, field, up, t, null);
        if (phi > bestPhi + SEAT_TOL) {
          bestPhi = phi;
          bestT = t;
        }
      }
      body.x[1] += bestT;
      if (bestPhi < -CONTACT_BIAS) report.flags.push('nested');
    }

    // Dynamics always run — soft depenetration makes them safe from any start — under
    // the wander budget: conform to the surface, but stay where the phrase said.
    const seated = { x: [...body.x], q: [...body.q] };
    const budget = Math.max(0.03, 0.15 * bDiag);
    const settled = settle(body, field, down.map((c) => c * GRAVITY), ground, budget);
    report.steps = settled.steps;
    report.flags.push(...settled.flags.filter((flag) => flag !== 'drifted'));

    if (settled.flags.includes('drifted')) {
      body.x = [...seated.x];
      body.q = [...seated.q];
      report.flags.push('settled_away');
    }
    pushToContact(body, field, grid, down, ground, 4 * field.h, 5);

    // Settled on the ground a sliver away from the anchor: the phrase put B against A,
    // so close a gap that small sideways rather than leave a visible seam.
    const lateral = measureGap(body, field, grid);
    if (lateral > SNAP_MIN && lateral < SNAP_MAX) {
      const toward = towardAnchor(body, field);
      const flat = toward ? unit([toward[0], 0, toward[2]]) : null;
      if (flat) pushToContact(body, field, grid, flat, null, 2 * SNAP_MAX, 8);
    }

    flagDroppedFar(body, field, report);
  } else {
    // lean: gravity tilted into the anchor, so the settled pose presses against the
    // face instead of sliding away from it. Unlike rest, the lateral travel is the
    // point — the body must reach both the ground and the face — so it keeps the
    // dynamic flow with a generous cap and its own completion checks.
    const toward = given ?? towardAnchor(body, field);
    const side = toward ? unit([toward[0], 0, toward[2]]) : null;
    const gravityDir = side ? unit([side[0] * LEAN_TILT, -1, side[2] * LEAN_TILT]) : down;
    if (!side) report.flags.push('lean_no_side');

    // A penetrating start backs out to a kiss before simulating, as a pure translation:
    // resolving the overlap through friction-bearing contacts would impart a spin and a
    // slide the phrase never asked for. `inward` points deeper into A, the approach
    // direction seat's back-out expects.
    if (gapBefore < -CONTACT_BIAS) {
      const inward = towardAnchor(body, field);
      if (inward) seat(body, field, inward, -CONTACT_BIAS, null, BACKOUT_CELLS * field.h);
    }

    const gravity = gravityDir.map((c) => c * GRAVITY);
    const settled = settle(body, field, gravity, ground, Math.max(0.25, 0.75 * bDiag));
    report.flags.push(...settled.flags);
    report.steps = settled.steps;

    if (settled.flags.includes('drifted')) {
      report.gap_after = report.gap_before;
      return { placed, report };
    }

    pushToContact(body, field, grid, down, ground, 4 * field.h, 5);

    if (side) {
      // Seal the face contact — or, when the tilt made the body slide clear of the face
      // altogether, push it home sideways. A face further than a body-scale slide away
      // was never the lean the model described, so that reverts and flags instead.
      const pushed = pushToContact(body, field, grid, side, null, 0.12, 16);
      if (!pushed.ok) report.flags.push('lean_incomplete');
    }

    flagDroppedFar(body, field, report);
  }

  report.gap_after = measureGap(body, field, grid) * aSize;
  report.moved = Math.hypot(
    body.x[0] - body.start[0],
    body.x[1] - body.start[1],
    body.x[2] - body.start[2],
  ) * aSize;
  report.rotated_degrees = Number(qAngleDeg(body.q).toFixed(2));

  // The solver worked on a delta about the sample centroid; fold it into the TRS. The
  // rotation composes on the left (it happened in world space), and the position carries
  // the centroid's own motion — all denormalized back to anchor units.
  const dq = body.q;
  const rotation = qMul(dq, placed.rotation, [0, 0, 0, 1]);
  const start = body.start.map((c) => c / norm);
  const end = body.x.map((c) => c / norm);
  const rel = [0, 0, 0];
  qRotate(dq, placed.position[0] - start[0], placed.position[1] - start[1], placed.position[2] - start[2], rel);

  return {
    placed: {
      position: [rel[0] + end[0], rel[1] + end[1], rel[2] + end[2]],
      rotation: [...qNormalize(rotation)],
      scale: [...placed.scale],
    },
    report,
  };
}

/**
 * The folder-based entry every seam uses: reads both raw meshes from `dir` and refines.
 * `objects` mirrors bake's shape — `[{ role, stem, trs }]` — so callers pass the same
 * structures they are about to bake.
 */
export function refineDir(dir, { objects, intent, options }) {
  const byRole = Object.fromEntries(objects.map((object) => [object.role, object]));
  const read = (stem) => {
    const data = fs.readFileSync(path.join(dir, `${stem}.glb`));
    return sceneTriangles(parseGLB(data));
  };
  return refinePlacement({
    anchorTriangles: read(byRole.anchor.stem),
    placedTriangles: read(byRole.placed.stem),
    anchor: byRole.anchor.trs,
    placed: byRole.placed.trs,
    intent,
    options,
  });
}
