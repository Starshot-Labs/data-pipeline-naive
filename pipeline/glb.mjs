// Minimal GLB reader/writer: enough to pull world-space triangles out of a
// glTF binary and to bake a rigid transform back into one.

import { Matrix4, Vector3, Quaternion } from 'three';

const MAGIC = 0x46546c67;
const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN = 0x004e4942;

const COMPONENT_SIZE = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 };
const NUM_COMPONENTS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

export function parseGLB(buffer) {
  if (buffer.readUInt32LE(0) !== MAGIC) throw new Error('not a GLB file');
  const total = buffer.readUInt32LE(8);
  let offset = 12;
  let json = null;
  let bin = null;
  while (offset + 8 <= total) {
    const length = buffer.readUInt32LE(offset);
    const type = buffer.readUInt32LE(offset + 4);
    const data = buffer.subarray(offset + 8, offset + 8 + length);
    if (type === CHUNK_JSON) json = JSON.parse(data.toString('utf8'));
    else if (type === CHUNK_BIN) bin = data;
    offset += 8 + length;
  }
  if (!json) throw new Error('GLB has no JSON chunk');
  return { json, bin };
}

function padTo4(buffer, fill) {
  const extra = (4 - (buffer.length % 4)) % 4;
  return extra ? Buffer.concat([buffer, Buffer.alloc(extra, fill)]) : buffer;
}

export function serializeGLB({ json, bin }) {
  const jsonChunk = padTo4(Buffer.from(JSON.stringify(json), 'utf8'), 0x20);
  const binChunk = bin ? padTo4(Buffer.from(bin), 0x00) : null;
  const total = 12 + 8 + jsonChunk.length + (binChunk ? 8 + binChunk.length : 0);

  const out = Buffer.alloc(total);
  out.writeUInt32LE(MAGIC, 0);
  out.writeUInt32LE(2, 4);
  out.writeUInt32LE(total, 8);
  out.writeUInt32LE(jsonChunk.length, 12);
  out.writeUInt32LE(CHUNK_JSON, 16);
  jsonChunk.copy(out, 20);
  if (binChunk) {
    const at = 20 + jsonChunk.length;
    out.writeUInt32LE(binChunk.length, at);
    out.writeUInt32LE(CHUNK_BIN, at + 4);
    binChunk.copy(out, at + 8);
  }
  return out;
}

function readComponent(view, at, componentType) {
  switch (componentType) {
    case 5120: return view.getInt8(at);
    case 5121: return view.getUint8(at);
    case 5122: return view.getInt16(at, true);
    case 5123: return view.getUint16(at, true);
    case 5125: return view.getUint32(at, true);
    case 5126: return view.getFloat32(at, true);
    default: throw new Error(`unsupported componentType ${componentType}`);
  }
}

export function readAccessor({ json, bin }, index) {
  const accessor = json.accessors[index];
  if (accessor.sparse) throw new Error('sparse accessors are not supported');
  if (accessor.bufferView === undefined) throw new Error('accessor without bufferView');

  const view = json.bufferViews[accessor.bufferView];
  const comps = NUM_COMPONENTS[accessor.type];
  const size = COMPONENT_SIZE[accessor.componentType];
  const stride = view.byteStride ?? comps * size;
  const base = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
  const data = new DataView(bin.buffer, bin.byteOffset, bin.byteLength);

  const out = new Float64Array(accessor.count * comps);
  for (let i = 0; i < accessor.count; i++) {
    for (let c = 0; c < comps; c++) {
      out[i * comps + c] = readComponent(data, base + i * stride + c * size, accessor.componentType);
    }
  }
  return out;
}

export function localMatrix(node) {
  const m = new Matrix4();
  if (node.matrix) return m.fromArray(node.matrix);
  return m.compose(
    new Vector3().fromArray(node.translation ?? [0, 0, 0]),
    new Quaternion().fromArray(node.rotation ?? [0, 0, 0, 1]),
    new Vector3().fromArray(node.scale ?? [1, 1, 1]),
  );
}

function collectPrimitive(glb, primitive, world, out) {
  if ((primitive.mode ?? 4) !== 4) return;
  if (primitive.extensions?.KHR_draco_mesh_compression) throw new Error('Draco-compressed meshes are not supported');
  const positionIndex = primitive.attributes?.POSITION;
  if (positionIndex === undefined) return;

  const positions = readAccessor(glb, positionIndex);
  const indices = primitive.indices !== undefined ? readAccessor(glb, primitive.indices) : null;
  const count = indices ? indices.length : positions.length / 3;
  const v = new Vector3();

  for (let i = 0; i + 2 < count; i += 3) {
    for (let k = 0; k < 3; k++) {
      const vertex = indices ? indices[i + k] : i + k;
      v.set(positions[vertex * 3], positions[vertex * 3 + 1], positions[vertex * 3 + 2]).applyMatrix4(world);
      out.push(v.x, v.y, v.z);
    }
  }
}

/** Every triangle of the default scene, flattened to 9 floats each, in the file's own coordinate frame. */
export function sceneTriangles(glb) {
  const { json } = glb;
  const scene = json.scenes?.[json.scene ?? 0];
  if (!scene) throw new Error('GLB has no scene');

  const out = [];
  const visit = (nodeIndex, parent) => {
    const node = json.nodes[nodeIndex];
    const world = parent.clone().multiply(localMatrix(node));
    if (node.mesh !== undefined) {
      for (const primitive of json.meshes[node.mesh].primitives) collectPrimitive(glb, primitive, world, out);
    }
    for (const child of node.children ?? []) visit(child, world);
  };
  for (const root of scene.nodes ?? []) visit(root, new Matrix4());

  if (!out.length) throw new Error('GLB has no triangles');
  return new Float64Array(out);
}

/**
 * Drop `alphaMode: BLEND` from solid meshes. Bakers tag a material BLEND whenever its
 * texture atlas carries an alpha channel — which it always does, for the padding around
 * the UV islands — and glTF loaders answer that by turning off depth writes, so the mesh
 * renders with no occlusion at all: backfaces, interiors and arbitrary triangle order.
 */
export function forceOpaqueMaterials(glb) {
  for (const material of glb.json.materials ?? []) delete material.alphaMode;
  return glb;
}

const PLACEMENT_NODE = 'placement';

/** Re-parent the scene under a single node carrying `trs`, baking the placement into the file. */
export function bakeTransform(glb, { position, rotation, scale }) {
  const json = structuredClone(glb.json);
  const scene = json.scenes[json.scene ?? 0];
  const roots = scene.nodes ?? [];
  if (!roots.length) throw new Error('GLB scene has no root nodes');

  json.nodes.push({ name: PLACEMENT_NODE, children: roots, translation: position, rotation, scale });
  scene.nodes = [json.nodes.length - 1];
  return { json, bin: glb.bin };
}

/**
 * Whether a placement has already been baked into this file. Meshes are baked in place, so
 * this is what stops a re-run from applying a second transform on top of the first — the
 * file itself carries the answer, which a crash between writing the mesh and writing the
 * metadata cannot desynchronise.
 */
export function isBaked({ json }) {
  const roots = json.scenes?.[json.scene ?? 0]?.nodes ?? [];
  return roots.length === 1 && json.nodes?.[roots[0]]?.name === PLACEMENT_NODE;
}

/**
 * Rewrites every vertex of the default scene through `map`, a function over positions in
 * the file's own scene frame — the same frame `sceneTriangles` speaks. This is how a
 * soft-body result becomes a file: the drape deforms world positions, and writing them
 * back through each node's inverse matrix keeps the glTF structure — nodes, materials,
 * UVs, the works — exactly as it was, with only the geometry moved.
 *
 * Normals are recomputed from the deformed triangles (area-weighted, so they come out of
 * the cross products already scaled by influence); tangents, if any, are left alone,
 * which shades a normal map slightly wrong on heavily bent regions — acceptable against
 * re-deriving a tangent basis. Buffers are rewritten in place at their original strides
 * and offsets, so shared and interleaved layouts survive.
 */
export function deformGLB(glb, map) {
  const json = structuredClone(glb.json);
  const bin = Buffer.from(glb.bin);
  const out = { json, bin };
  const scene = json.scenes?.[json.scene ?? 0];
  if (!scene) throw new Error('GLB has no scene');

  const seen = new Map(); // accessor index → matrix it was deformed under
  const write = (accessorIndex, values) => {
    const accessor = json.accessors[accessorIndex];
    const view = json.bufferViews[accessor.bufferView];
    const stride = view.byteStride ?? 12;
    const base = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
    for (let i = 0; i < accessor.count; i++) {
      for (let c = 0; c < 3; c++) bin.writeFloatLE(values[i * 3 + c], base + i * stride + c * 4);
    }
  };

  const visit = (nodeIndex, parent) => {
    const node = json.nodes[nodeIndex];
    const world = parent.clone().multiply(localMatrix(node));
    if (node.mesh !== undefined) deformMesh(out, node.mesh, world, map, seen, write);
    for (const child of node.children ?? []) visit(child, world);
  };
  for (const root of scene.nodes ?? []) visit(root, new Matrix4());

  return out;
}

function deformMesh(glb, meshIndex, world, map, seen, write) {
  const { json } = glb;
  const inverse = world.clone().invert();
  const v = new Vector3();

  for (const primitive of json.meshes[meshIndex].primitives) {
    const positionIndex = primitive.attributes?.POSITION;
    if (positionIndex === undefined) continue;
    if (seen.has(positionIndex)) {
      if (!seen.get(positionIndex).equals(world)) {
        throw new Error('a POSITION accessor is shared across differently-posed nodes — cannot deform');
      }
      continue;
    }
    seen.set(positionIndex, world.clone());

    const positions = readAccessor(glb, positionIndex);
    const count = positions.length / 3;
    const deformed = new Float64Array(positions.length);
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < count; i++) {
      v.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]).applyMatrix4(world);
      const p = map(v.x, v.y, v.z);
      v.set(p[0], p[1], p[2]).applyMatrix4(inverse);
      deformed[i * 3] = v.x;
      deformed[i * 3 + 1] = v.y;
      deformed[i * 3 + 2] = v.z;
      for (let c = 0; c < 3; c++) {
        const value = v.getComponent(c);
        if (value < min[c]) min[c] = value;
        if (value > max[c]) max[c] = value;
      }
    }
    write(positionIndex, deformed);
    json.accessors[positionIndex].min = min;
    json.accessors[positionIndex].max = max;

    const normalIndex = primitive.attributes?.NORMAL;
    if (normalIndex !== undefined && !seen.has(normalIndex)) {
      seen.set(normalIndex, world.clone());
      write(normalIndex, recomputeNormals(glb, primitive, deformed));
    }
  }
}

/** Area-weighted vertex normals of a primitive's deformed geometry, in node-local space. */
function recomputeNormals(glb, primitive, positions) {
  const indices = primitive.indices !== undefined ? readAccessor(glb, primitive.indices) : null;
  const count = indices ? indices.length : positions.length / 3;
  const normals = new Float64Array(positions.length);

  for (let i = 0; i + 2 < count; i += 3) {
    const a = indices ? indices[i] : i;
    const b = indices ? indices[i + 1] : i + 1;
    const c = indices ? indices[i + 2] : i + 2;
    const ux = positions[b * 3] - positions[a * 3];
    const uy = positions[b * 3 + 1] - positions[a * 3 + 1];
    const uz = positions[b * 3 + 2] - positions[a * 3 + 2];
    const vx = positions[c * 3] - positions[a * 3];
    const vy = positions[c * 3 + 1] - positions[a * 3 + 1];
    const vz = positions[c * 3 + 2] - positions[a * 3 + 2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    for (const vertex of [a, b, c]) {
      normals[vertex * 3] += nx;
      normals[vertex * 3 + 1] += ny;
      normals[vertex * 3 + 2] += nz;
    }
  }
  for (let i = 0; i < normals.length; i += 3) {
    const len = Math.hypot(normals[i], normals[i + 1], normals[i + 2]) || 1;
    normals[i] /= len;
    normals[i + 1] /= len;
    normals[i + 2] /= len;
  }
  return normals;
}

/**
 * The TRS a baked file carries, or null for a raw mesh. The inverse of `bakeTransform`:
 * poses live in the files and nowhere else, so anything that wants to continue from the
 * current arrangement — a physics-only pass, most notably — starts by reading it back.
 */
export function extractTransform({ json }) {
  const roots = json.scenes?.[json.scene ?? 0]?.nodes ?? [];
  const node = roots.length === 1 ? json.nodes?.[roots[0]] : null;
  if (!node || node.name !== PLACEMENT_NODE) return null;
  return {
    position: node.translation ?? [0, 0, 0],
    rotation: node.rotation ?? [0, 0, 0, 1],
    scale: node.scale ?? [1, 1, 1],
  };
}
