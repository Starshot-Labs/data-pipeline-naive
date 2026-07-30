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

export function rasterize(triangles, dims, voxelSize, gridMin) {
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

export function floodOutside(data, dims) {
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

export function voxelize(triangles, resolution) {
  const { min, max } = bounds(triangles);
  const size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];

  const voxelSize = Math.max(...size) / resolution;
  if (!(voxelSize > 0)) throw new Error('degenerate mesh bounds');

  const dims = size.map((s) => Math.max(1, Math.ceil(s / voxelSize)));
  const origin = dims.map((d) => (-d * voxelSize) / 2);
  const gridMin = [center[0] + origin[0], center[1] + origin[1], center[2] + origin[2]];

  const fineDims = dims.map((d) => d * SUBDIVISION);
  const fine = rasterize(triangles, fineDims, voxelSize / SUBDIVISION, gridMin);
  floodOutside(fine, fineDims);

  return { dims, voxelSize, origin, center, size, data: downsample(fine, fineDims, dims) };
}

const ORDERS = [
  [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0],
];
const OTHERS = [[1, 2], [0, 2], [0, 1]];

/** A max-heap of candidate boxes keyed by volume, small enough to live here. */
class Heap {
  items = [];
  get size() { return this.items.length; }
  peek() { return this.items[0]; }
  push(item) {
    const a = this.items;
    let i = a.push(item) - 1;
    for (let p; i > 0 && a[(p = (i - 1) >> 1)].volume < a[i].volume; i = p) [a[p], a[i]] = [a[i], a[p]];
  }
  pop() {
    const a = this.items;
    const top = a[0];
    const last = a.pop();
    if (a.length) {
      a[0] = last;
      for (let i = 0; ; ) {
        let largest = i;
        for (const c of [2 * i + 1, 2 * i + 2]) if (c < a.length && a[c].volume > a[largest].volume) largest = c;
        if (largest === i) break;
        [a[largest], a[i]] = [a[i], a[largest]];
        i = largest;
      }
    }
    return top;
  }
}

/**
 * Decomposes the solid voxels into disjoint axis-aligned blocks, extracted largest first so
 * every block is as big as the remaining volume allows: a solid cube is one block, a cube
 * with a thick slice inset into it is three. Extracting in seed order instead would let a
 * box grown from a narrow region leak through a constriction and fragment the larger volume
 * behind it.
 *
 * Candidates grow greedily from region corners (voxels with nothing free on their three
 * negative sides) along all six axis orders, the largest of the six winning. The heap is
 * lazy: a popped candidate is regrown against current coverage and emitted only if it still
 * beats what remains, so nothing is recomputed until the moment it might actually win.
 * Emitting a block can only create new corners on its three positive faces, which is what
 * keeps the candidate set complete; the lexicographic-minimum sweep is the backstop that
 * guarantees a corner always exists while anything is uncovered.
 *
 * Blocks come back as `[x0, y0, z0, x1, y1, z1]` in the object's own bbox-centred frame
 * (world units), sorted largest first, rounded just past voxel precision so adjacent block
 * faces stay distinct without dragging float noise into the prompt.
 */
export function toBlocks({ dims, data, voxelSize, origin }) {
  const [dx, dy, dz] = dims;
  const covered = new Uint8Array(data.length);
  const free = (x, y, z) =>
    x >= 0 && y >= 0 && z >= 0 && x < dx && y < dy && z < dz &&
    data[(y * dz + z) * dx + x] === 1 && !covered[(y * dz + z) * dx + x];
  const isCorner = (x, y, z) =>
    free(x, y, z) && !free(x - 1, y, z) && !free(x, y - 1, z) && !free(x, y, z - 1);

  // Whether the whole face of `extent` at `min`, one step further along `axis`, is free.
  const growable = (min, extent, axis) => {
    const edge = min[axis] + extent[axis];
    if (edge >= dims[axis]) return false;
    const [u, v] = OTHERS[axis];
    const p = [0, 0, 0];
    p[axis] = edge;
    for (let i = 0; i < extent[u]; i++) {
      p[u] = min[u] + i;
      for (let j = 0; j < extent[v]; j++) {
        p[v] = min[v] + j;
        if (!free(p[0], p[1], p[2])) return false;
      }
    }
    return true;
  };

  const grow = (x, y, z) => {
    const min = [x, y, z];
    let best = null;
    for (const order of ORDERS) {
      const extent = [1, 1, 1];
      for (const axis of order) while (growable(min, extent, axis)) extent[axis]++;
      const volume = extent[0] * extent[1] * extent[2];
      if (!best || volume > best.volume) best = { min, extent, volume };
    }
    return best;
  };

  const heap = new Heap();
  const seed = (x, y, z) => { if (isCorner(x, y, z)) heap.push(grow(x, y, z)); };
  const sweep = () => {
    for (let y = 0; y < dy; y++)
      for (let z = 0; z < dz; z++)
        for (let x = 0; x < dx; x++) seed(x, y, z);
  };
  sweep();

  let remaining = 0;
  for (const cell of data) remaining += cell;

  const blocks = [];
  while (remaining > 0) {
    if (!heap.size) { sweep(); continue; }

    const popped = heap.pop();
    if (!free(...popped.min)) continue;
    const box = grow(...popped.min);
    if (heap.size && box.volume < heap.peek().volume) {
      heap.push(box);
      continue;
    }

    const [x, y, z] = box.min;
    const [ex, ey, ez] = box.extent;
    for (let by = y; by < y + ey; by++)
      for (let bz = z; bz < z + ez; bz++)
        for (let bx = x; bx < x + ex; bx++) covered[(by * dz + bz) * dx + bx] = 1;
    remaining -= box.volume;
    blocks.push(box);

    // Coverage only turns cells from free to covered, so the only voxels whose corner
    // status can change are the ones just past the block's three positive faces.
    for (let by = y; by < y + ey; by++) for (let bz = z; bz < z + ez; bz++) seed(x + ex, by, bz);
    for (let bx = x; bx < x + ex; bx++) for (let bz = z; bz < z + ez; bz++) seed(bx, y + ey, bz);
    for (let bx = x; bx < x + ex; bx++) for (let by = y; by < y + ey; by++) seed(bx, by, z + ez);
  }

  const decimals = Math.max(0, Math.ceil(-Math.log10(voxelSize)) + 1);
  const world = (c, i) => Number((origin[c] + i * voxelSize).toFixed(decimals));
  blocks.sort((a, b) => b.volume - a.volume);
  return blocks.map(({ min, extent }) => [
    world(0, min[0]), world(1, min[1]), world(2, min[2]),
    world(0, min[0] + extent[0]), world(1, min[1] + extent[1]), world(2, min[2] + extent[2]),
  ]);
}
