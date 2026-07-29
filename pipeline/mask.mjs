// The one input VoxHammer needs that a placement sample does not have: a mask mesh
// marking the region of the scene that editing may touch.
//
// It is derived rather than authored. An LLM reads the anchor's occupancy grid and the
// placement phrase and returns one broad axis-aligned box of cells — everywhere the
// object it has not seen yet might plausibly generate. The anchor's own voxels are
// subtracted from that box, so the region offered to the editor is exactly the empty
// space around the part of the anchor the phrase names, and never the anchor itself.
//
// Two things about the frame matter, both forced by VoxHammer:
//
//   · `delete_region_voxel.py` imports the mask GLB with no normalization and tests it
//     against `preset_grid64.ply`, a fixed 64³ lattice over [-0.5, 0.5]³. So the mask
//     has to be authored in that cube, which is also why the anchor is baked into it
//     (`unitCube`) instead of being left for Blender to normalize.
//   · the inside/outside test is `pysdf`, so the mask has to be watertight. The mesh
//     here is the filled/empty boundary with welded corners and outward winding, and
//     its signed volume is checked against the voxel count it should enclose.

import { serializeGLB } from './glb.mjs';
import { voxelizeLattice, bounds } from './voxelize.mjs';
import { describe } from './place.mjs';
import { chatJSON } from './openrouter.mjs';

/** TRELLIS's sparse-structure resolution: what `voxels.ply` and the mask are both read on. */
export const GRID = 64;

const VOXEL = 1 / GRID;
const CUBE_MIN = [-0.5, -0.5, -0.5];

const at = (x, y, z) => (y * GRID + z) * GRID + x;
const clamp = (v) => Math.min(GRID - 1, Math.max(0, Math.round(v)));

/** The TRS that lands a mesh in TRELLIS's cube: longest axis 1, bounding-box centre at the origin. */
export function unitCube(triangles) {
  const { min, max } = bounds(triangles);
  const size = [0, 1, 2].map((c) => max[c] - min[c]);
  const scale = 1 / Math.max(...size);
  if (!Number.isFinite(scale) || scale <= 0) throw new Error('degenerate mesh bounds');
  const centre = [0, 1, 2].map((c) => (min[c] + max[c]) / 2);
  return { position: centre.map((v) => -v * scale), rotation: [0, 0, 0, 1], scale: [scale, scale, scale] };
}

/** Solid occupancy of an already-normalized mesh on the 64³ lattice. */
export const lattice = (triangles) =>
  voxelizeLattice(triangles, { dims: [GRID, GRID, GRID], voxelSize: VOXEL, gridMin: CUBE_MIN });

// ── the box ──────────────────────────────────────────────────────────────────

const SYSTEM = `You are preparing a 3D editing job. You get the voxel occupancy grid of an anchor object A that already exists, and a phrase describing an object B that is about to be generated and placed relative to A. Return the axis-aligned box of grid cells that B might occupy.

The grid is the generator's own lattice: exactly 64 x 64 x 64 cells over the cube A was normalized into, so A's longest axis spans all 64 cells and its shorter axes leave empty cells on either side of it. Indices are integers 0..63 on every axis — i along +X, j along +Y (up), k along +Z (A's front face), right-handed.

Reading a grid: it is printed as horizontal slices, top layer first. Inside a slice each line is one Z index (first line z=0, last line z=dz-1) and each character is one X index starting at x=0; '#' is solid, '.' is empty. Cells are cubes, so the grid's proportions are A's real proportions. Consecutive layers that are identical are printed once under a range header like \`y=31..12\`, meaning every layer from 31 down to 12 looks like that.

Answer with:
- reasoning: locate the surface, cavity or face the phrase names by reading A's slices, state which cell ranges it spans, then size the region B needs there. Do the arithmetic here.
- min: [i, j, k] of the lowest-indexed cell in the box
- max: [i, j, k] of the highest-indexed cell in the box, inclusive

This is a generation region, not a fit. B does not exist yet — its exact size, shape and pose are unknown — so the box must contain every cell B could plausibly occupy plus the slack it needs to look right, and it must reach the surface B rests against or the cavity it sits in. Err on the large side: a box that is too tight clips B or leaves it nowhere to sit, while a box that is too broad only costs precision. It should still be targeted rather than the whole grid — the region the phrase points at, generously bounded, not the entire cube.`;

const SCHEMA = {
  type: 'object',
  properties: {
    reasoning: { type: 'string' },
    min: { type: 'array', items: { type: 'number' } },
    max: { type: 'array', items: { type: 'number' } },
  },
  required: ['reasoning', 'min', 'max'],
  additionalProperties: false,
};

export const buildBoxPrompt = ({ phrase, object, anchor }) =>
  [`PLACEMENT: "${phrase}"`, `OBJECT TO GENERATE: ${object}`, '', describe('A (anchor, already generated)', anchor)].join('\n');

/** The LLM's box, rounded onto the lattice and clamped into it. */
export async function requestBox({ phrase, object, anchor, model }) {
  const { data, model: used, usage } = await chatJSON({
    model,
    system: SYSTEM,
    user: buildBoxPrompt({ phrase, object, anchor }),
    name: 'mask_box',
    schema: SCHEMA,
  });

  const corners = [data.min, data.max];
  if (!corners.every((corner) => Array.isArray(corner) && corner.length === 3 && corner.every(Number.isFinite))) {
    throw new Error(`invalid box from model: ${JSON.stringify(data).slice(0, 500)}`);
  }

  // The model is answering about a region, so an inverted or out-of-range corner is worth
  // straightening rather than re-rolling; a box that survives that and is still empty is not.
  const low = data.min.map(clamp);
  const high = data.max.map(clamp);
  const min = [0, 1, 2].map((c) => Math.min(low[c], high[c]));
  const max = [0, 1, 2].map((c) => Math.max(low[c], high[c]));

  return { reasoning: data.reasoning, min, max, model: used, usage };
}

// ── the mask ─────────────────────────────────────────────────────────────────

/** The box minus the anchor: the cells editing is allowed to fill. */
export function carve({ min, max }, anchor) {
  const data = new Uint8Array(GRID ** 3);
  let count = 0;
  for (let y = min[1]; y <= max[1]; y++) {
    for (let z = min[2]; z <= max[2]; z++) {
      for (let x = min[0]; x <= max[0]; x++) {
        const index = at(x, y, z);
        if (anchor[index]) continue;
        data[index] = 1;
        count++;
      }
    }
  }
  if (!count) throw new Error('the box is entirely inside the anchor — nothing left to edit');
  return { data, count };
}

// The two axes that complete each axis in cyclic order. Taking a face's corners as
// (b,c) = (0,0), (1,0), (1,1), (0,1) then winds it along +axis by the right-hand rule.
const OTHER_AXES = [[1, 2], [2, 0], [0, 1]];

function faceCorners(cell, axis, side) {
  const [b, c] = OTHER_AXES[axis];
  const quad = [[0, 0], [1, 0], [1, 1], [0, 1]].map(([db, dc]) => {
    const corner = [0, 0, 0];
    corner[axis] = cell[axis] + (side > 0 ? 1 : 0);
    corner[b] = cell[b] + db;
    corner[c] = cell[c] + dc;
    return corner;
  });
  return side > 0 ? quad : quad.reverse();
}

/**
 * The filled/empty boundary as a welded, outward-wound triangle mesh. Corners are keyed on
 * their integer lattice position, so neighbouring faces share vertices exactly and the
 * result is closed rather than merely gap-free.
 */
export function surfaceMesh(data) {
  const corners = new Map();
  const positions = [];
  const indices = [];

  const vertexOf = (corner) => {
    const key = (corner[1] * (GRID + 1) + corner[2]) * (GRID + 1) + corner[0];
    let index = corners.get(key);
    if (index === undefined) {
      index = positions.length / 3;
      corners.set(key, index);
      for (const c of [0, 1, 2]) positions.push(CUBE_MIN[c] + corner[c] * VOXEL);
    }
    return index;
  };

  const filled = (x, y, z) =>
    x >= 0 && y >= 0 && z >= 0 && x < GRID && y < GRID && z < GRID && data[at(x, y, z)];

  for (let y = 0; y < GRID; y++) {
    for (let z = 0; z < GRID; z++) {
      for (let x = 0; x < GRID; x++) {
        if (!data[at(x, y, z)]) continue;
        for (const axis of [0, 1, 2]) {
          for (const side of [-1, 1]) {
            const neighbour = [x, y, z];
            neighbour[axis] += side;
            if (filled(...neighbour)) continue;
            const [a, b, c, d] = faceCorners([x, y, z], axis, side).map(vertexOf);
            indices.push(a, b, c, a, c, d);
          }
        }
      }
    }
  }

  return { positions: new Float32Array(positions), indices: new Uint32Array(indices) };
}

/**
 * The volume the shell encloses, by the divergence theorem. Outward winding makes it
 * positive, and it comes to exactly one cell per filled voxel — which is the cheapest
 * check that the mesh is both closed and wound the right way round.
 */
export function enclosedVolume({ positions, indices }) {
  let total = 0;
  const point = (i) => [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];
  for (let i = 0; i < indices.length; i += 3) {
    const [ax, ay, az] = point(indices[i]);
    const [bx, by, bz] = point(indices[i + 1]);
    const [cx, cy, cz] = point(indices[i + 2]);
    total += ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
  }
  return total / 6;
}

/** A GLB holding one untextured mesh, which is all a mask has to be. */
export function meshToGLB({ positions, indices }) {
  const vertices = Buffer.from(positions.buffer, positions.byteOffset, positions.byteLength);
  const elements = Buffer.from(indices.buffer, indices.byteOffset, indices.byteLength);

  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i++) {
    const c = i % 3;
    if (positions[i] < min[c]) min[c] = positions[i];
    if (positions[i] > max[c]) max[c] = positions[i];
  }

  return serializeGLB({
    json: {
      asset: { version: '2.0', generator: 'data-creator/mask' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ name: 'mask', mesh: 0 }],
      meshes: [{ name: 'mask', primitives: [{ attributes: { POSITION: 0 }, indices: 1, material: 0 }] }],
      materials: [{ name: 'mask', pbrMetallicRoughness: { baseColorFactor: [0.9, 0.2, 0.2, 1], metallicFactor: 0, roughnessFactor: 1 } }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: positions.length / 3, type: 'VEC3', min, max },
        { bufferView: 1, componentType: 5125, count: indices.length, type: 'SCALAR' },
      ],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: vertices.length, target: 34962 },
        { buffer: 0, byteOffset: vertices.length, byteLength: elements.length, target: 34963 },
      ],
      buffers: [{ byteLength: vertices.length + elements.length }],
    },
    bin: Buffer.concat([vertices, elements]),
  });
}

/**
 * Box → mask, with the geometry checked against the voxels it came from. `anchor` is the
 * occupancy from `lattice`, and the returned GLB is already in the cube VoxHammer reads.
 */
export function buildMask(box, anchor) {
  const { data, count } = carve(box, anchor);
  const mesh = surfaceMesh(data);
  const volume = enclosedVolume(mesh);
  const expected = count * VOXEL ** 3;
  if (Math.abs(volume - expected) > expected * 1e-3) {
    throw new Error(`mask shell encloses ${volume.toExponential(3)}, expected ${expected.toExponential(3)}`);
  }
  return { glb: meshToGLB(mesh), data, voxels: count, triangles: mesh.indices.length / 3, volume };
}
