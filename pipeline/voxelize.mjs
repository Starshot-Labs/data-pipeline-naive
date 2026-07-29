// Surface-sample voxelization with an outside-in flood fill so closed volumes
// come out solid. Grids are cubic-celled and centred on the mesh bounding box,
// so `origin` is expressed in the object's own bbox-centred frame.
//
// Sampling and filling happen on a sub-divided grid; an output voxel only counts
// as occupied once more than MIN_FILL of its volume is solid, which drops the
// slivers a surface leaves when it clips a corner.

const SAMPLES_PER_VOXEL = 2;
const SUBDIVISION = 3;
const MIN_FILL = 0.2;

export function bounds(triangles) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < triangles.length; i += 3) {
    for (let c = 0; c < 3; c++) {
      const v = triangles[i + c];
      if (v < min[c]) min[c] = v;
      if (v > max[c]) max[c] = v;
    }
  }
  return { min, max };
}

function rasterize(triangles, dims, voxelSize, gridMin) {
  const data = new Uint8Array(dims[0] * dims[1] * dims[2]);
  const mark = (x, y, z) => {
    const ix = Math.min(dims[0] - 1, Math.max(0, Math.floor((x - gridMin[0]) / voxelSize)));
    const iy = Math.min(dims[1] - 1, Math.max(0, Math.floor((y - gridMin[1]) / voxelSize)));
    const iz = Math.min(dims[2] - 1, Math.max(0, Math.floor((z - gridMin[2]) / voxelSize)));
    data[(iy * dims[2] + iz) * dims[0] + ix] = 1;
  };

  const step = voxelSize / SAMPLES_PER_VOXEL;
  for (let t = 0; t < triangles.length; t += 9) {
    const ax = triangles[t], ay = triangles[t + 1], az = triangles[t + 2];
    const ux = triangles[t + 3] - ax, uy = triangles[t + 4] - ay, uz = triangles[t + 5] - az;
    const vx = triangles[t + 6] - ax, vy = triangles[t + 7] - ay, vz = triangles[t + 8] - az;
    const edges = Math.max(
      Math.hypot(ux, uy, uz),
      Math.hypot(vx, vy, vz),
      Math.hypot(vx - ux, vy - uy, vz - uz),
    );
    const n = Math.max(1, Math.ceil(edges / step));
    for (let i = 0; i <= n; i++) {
      for (let j = 0; i + j <= n; j++) {
        const u = i / n;
        const v = j / n;
        mark(ax + ux * u + vx * v, ay + uy * u + vy * v, az + uz * u + vz * v);
      }
    }
  }
  return data;
}

function floodOutside(data, dims) {
  const [dx, dy, dz] = dims;
  const outside = new Uint8Array(data.length);
  const stack = new Int32Array(data.length);
  let top = 0;

  const seed = (x, y, z) => {
    const i = (y * dz + z) * dx + x;
    if (data[i] || outside[i]) return;
    outside[i] = 1;
    stack[top++] = i;
  };

  for (let y = 0; y < dy; y++)
    for (let z = 0; z < dz; z++)
      for (let x = 0; x < dx; x++)
        if (x === 0 || x === dx - 1 || y === 0 || y === dy - 1 || z === 0 || z === dz - 1) seed(x, y, z);

  while (top) {
    const i = stack[--top];
    const x = i % dx;
    const rest = (i - x) / dx;
    const z = rest % dz;
    const y = (rest - z) / dz;
    if (x > 0) seed(x - 1, y, z);
    if (x < dx - 1) seed(x + 1, y, z);
    if (y > 0) seed(x, y - 1, z);
    if (y < dy - 1) seed(x, y + 1, z);
    if (z > 0) seed(x, y, z - 1);
    if (z < dz - 1) seed(x, y, z + 1);
  }

  for (let i = 0; i < data.length; i++) if (!outside[i]) data[i] = 1;
}

function downsample(fine, fineDims, dims) {
  const [fx, , fz] = fineDims;
  const [dx, dy, dz] = dims;
  const minCells = Math.floor(SUBDIVISION ** 3 * MIN_FILL) + 1;
  const data = new Uint8Array(dx * dy * dz);

  for (let y = 0; y < dy; y++) {
    for (let z = 0; z < dz; z++) {
      for (let x = 0; x < dx; x++) {
        let filled = 0;
        for (let sy = 0; sy < SUBDIVISION; sy++) {
          for (let sz = 0; sz < SUBDIVISION; sz++) {
            const row = ((y * SUBDIVISION + sy) * fz + z * SUBDIVISION + sz) * fx + x * SUBDIVISION;
            for (let sx = 0; sx < SUBDIVISION; sx++) filled += fine[row + sx];
          }
        }
        if (filled >= minCells) data[(y * dz + z) * dx + x] = 1;
      }
    }
  }
  return data;
}

/**
 * Occupancy on a lattice the caller picks: `dims` cubic cells of `voxelSize` with the
 * grid's minimum corner at `gridMin`. For when the grid has to line up with something
 * else's — a fixed 64³ cube, say — rather than with the mesh's own bounds.
 */
export function voxelizeLattice(triangles, { dims, voxelSize, gridMin }) {
  const fineDims = dims.map((d) => d * SUBDIVISION);
  const fine = rasterize(triangles, fineDims, voxelSize / SUBDIVISION, gridMin);
  floodOutside(fine, fineDims);
  return downsample(fine, fineDims, dims);
}

export function voxelize(triangles, resolution) {
  const { min, max } = bounds(triangles);
  const size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];

  const voxelSize = Math.max(...size) / resolution;
  if (!(voxelSize > 0)) throw new Error('degenerate mesh bounds');

  const dims = size.map((s) => Math.max(1, Math.ceil(s / voxelSize)));
  const origin = dims.map((d) => (-d * voxelSize) / 2);
  const gridMin = [center[0] + origin[0], center[1] + origin[1], center[2] + origin[2]];

  return { dims, voxelSize, origin, center, size, data: voxelizeLattice(triangles, { dims, voxelSize, gridMin }) };
}

/** One string per Y layer, indexed bottom-up. Each layer has `dz` rows of `dx` characters. */
export function toSlices({ dims, data }) {
  const [dx, dy, dz] = dims;
  const slices = [];
  for (let y = 0; y < dy; y++) {
    const rows = [];
    for (let z = 0; z < dz; z++) {
      let row = '';
      for (let x = 0; x < dx; x++) row += data[(y * dz + z) * dx + x] ? '#' : '.';
      rows.push(row);
    }
    slices.push(rows.join('\n'));
  }
  return slices;
}
