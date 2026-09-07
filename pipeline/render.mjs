// A software renderer for the per-object reference images: one GLB in, one PNG out.
//
// Pure JS on purpose. Blender or a GPU context would render prettier, but this has to run
// wherever the pipeline runs — a Modal CPU container, a laptop, inside dc-scene-ops — with
// zero native or system dependencies beyond what the pipeline already carries (`three` for
// the math, `sharp` to decode textures and encode the PNG). "Basic rendering" is the spec:
// textured, sun-lit, shadowed, and fast; not path-traced.
//
// The look: the object on solid black, centred on its own projected silhouette rather than
// on its bounding-box centre — under perspective those are not the same point, and a capture
// wants the pixels centred. Seen from its glTF front (+Z) with the same 10° downward pitch
// the manual exporter uses, lit by one directional sun plus a sky ambient. The shadow map is
// kept for self-shadowing, which is what gives the form depth; nothing is cast onto a ground
// plane, and there is no ground plane to cast onto.
//
// Rendered at 2× and downsampled, which is the cheapest anti-aliasing there is.

import sharp from 'sharp';
import { Matrix3, Matrix4, Vector3 } from 'three';
import { readAccessor, localMatrix } from './glb.mjs';

/**
 * Bumped whenever a change here alters the pixels.
 *
 * Renders are cached per uid and copied into every sample that seeds from that asset, so
 * without a version in the key a new renderer reaches only assets nobody has fetched before
 * — a corpus that has already been through the fetch keeps the picture the old one made, and
 * the change looks like it did nothing. This is what makes the cache miss on purpose.
 *
 * v2: camera-side fill, lifted ambient, and a black floor so no drawn pixel is pure black.
 */
export const RENDER_VERSION = 'v2';

const SIZE = () => Number(process.env.RENDER_SIZE ?? 512);
const FOV = (35 * Math.PI) / 180;
const PITCH = (10 * Math.PI) / 180;
const SHADOW_RES = 1024;
// Down-and-inward from the upper front-left, so form shows on the front faces the camera sees.
const SUN = new Vector3(-0.45, -1, -0.35).normalize();

// Ambient sky, the sun, and a fill from the camera's side.
//
// The fill is the term that makes a dark object legible at all. The background is solid
// black, so a low-albedo surface lit only by ambient lands within a couple of values of it
// and the silhouette disappears — and a reference image whose subject cannot be made out is
// worth nothing to the model reading it. A headlight term gives every surface the camera can
// see some light in proportion to how squarely it faces us, which separates the object from
// the background without flattening the sun's modelling.
//
// Kept just under a peak of 1 for a white surface facing both lights, so brightening the
// darks does not clip the brights. `RENDER_EXPOSURE` is a final gain over the lot, for
// lifting a whole corpus without re-tuning three terms against each other.
const AMBIENT = 0.42;
const SUN_STRENGTH = 0.55;
const FILL = 0.24;
// How much of the ambient a fully shadowed surface loses. Lower than it was: crushing
// self-shadowed areas to near-black is the same legibility problem in miniature.
const SHADOW_DEPTH = 0.7;
const EXPOSURE = () => Number(process.env.RENDER_EXPOSURE ?? 1);

// No pixel the object covers may come out pure black.
//
// Lighting is multiplicative, so an albedo of zero stays zero however bright the lamps get,
// and a black surface on a black background is a hole in the silhouette rather than a dark
// part of it. Every drawn fragment is therefore remapped from [0, 1] onto [FLOOR, 1]: white
// stays white, the ordering of everything between is preserved, and the darkest possible
// surface lands at a value that is unmistakably black to look at and unmistakably not the
// background to a reader. Only fragments go through it, so the background stays a true zero.
const BLACK_FLOOR = () => Number(process.env.RENDER_BLACK_FLOOR ?? 0.06);

// Normalized integer attributes (common for COLOR_0 and compressed UVs) carry a divisor.
const NORMALIZE = { 5120: 127, 5121: 255, 5122: 32767, 5123: 65535 };

function attribute(glb, index) {
  if (index === undefined) return null;
  const values = readAccessor(glb, index);
  const accessor = glb.json.accessors[index];
  const divisor = accessor.normalized ? NORMALIZE[accessor.componentType] : null;
  if (divisor) for (let i = 0; i < values.length; i++) values[i] = Math.max(values[i] / divisor, -1);
  return values;
}

/** baseColor factor + texture for a material, reading the spec-gloss extension too — a lot
 *  of Sketchfab-era assets still carry their colour there rather than in metallic-rough. */
function materialOf(json, index) {
  const material = index === undefined ? {} : (json.materials?.[index] ?? {});
  const specGloss = material.extensions?.KHR_materials_pbrSpecularGlossiness;
  const pbr = material.pbrMetallicRoughness ?? {};
  const factor = specGloss?.diffuseFactor ?? pbr.baseColorFactor ?? [1, 1, 1, 1];
  const textureInfo = specGloss?.diffuseTexture ?? pbr.baseColorTexture ?? null;
  const transform = textureInfo?.extensions?.KHR_texture_transform ?? null;
  return {
    factor,
    image: textureInfo ? (json.textures?.[textureInfo.index]?.source ?? null) : null,
    uvOffset: transform?.offset ?? [0, 0],
    uvScale: transform?.scale ?? [1, 1],
    // BLEND is treated as a cutout against the texture's alpha: correct sorting is not worth
    // its cost here, and the shader ignores the constant factor so glass stays visible.
    alphaCutoff: material.alphaMode && material.alphaMode !== 'OPAQUE' ? (material.alphaCutoff ?? 0.5) : null,
  };
}

/** Decodes every texture image the draw list references, once, to raw RGBA. */
async function decodeImages(glb, draws) {
  const wanted = new Set(draws.map((d) => d.material.image).filter((i) => i !== null));
  const images = new Map();
  for (const index of wanted) {
    const image = glb.json.images?.[index];
    let bytes = null;
    if (image?.bufferView !== undefined) {
      const view = glb.json.bufferViews[image.bufferView];
      bytes = glb.bin.subarray(view.byteOffset ?? 0, (view.byteOffset ?? 0) + view.byteLength);
    } else if (image?.uri?.startsWith('data:')) {
      bytes = Buffer.from(image.uri.slice(image.uri.indexOf(',') + 1), 'base64');
    }
    if (!bytes) continue; // externally-referenced image a GLB cannot carry — factor colour stands in
    try {
      const { data, info } = await sharp(bytes).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
      images.set(index, { data, width: info.width, height: info.height });
    } catch {
      // an undecodable texture only costs its colour, not the render
    }
  }
  return images;
}

/** Every renderable primitive of the default scene, with world-space geometry. */
function gatherDraws(glb) {
  const { json } = glb;
  const scene = json.scenes?.[json.scene ?? 0];
  if (!scene) throw new Error('GLB has no scene');

  const draws = [];
  const visit = (nodeIndex, parent) => {
    const node = json.nodes[nodeIndex];
    const world = parent.clone().multiply(localMatrix(node));
    if (node.mesh !== undefined) {
      for (const primitive of json.meshes[node.mesh].primitives) {
        if ((primitive.mode ?? 4) !== 4) continue;
        if (primitive.extensions?.KHR_draco_mesh_compression) throw new Error('Draco-compressed meshes are not supported');
        if (primitive.attributes?.POSITION === undefined) continue;

        const positions = attribute(glb, primitive.attributes.POSITION);
        const v = new Vector3();
        for (let i = 0; i < positions.length; i += 3) {
          v.set(positions[i], positions[i + 1], positions[i + 2]).applyMatrix4(world);
          positions[i] = v.x;
          positions[i + 1] = v.y;
          positions[i + 2] = v.z;
        }

        const normals = attribute(glb, primitive.attributes.NORMAL);
        if (normals) {
          const normalMatrix = new Matrix3().getNormalMatrix(world);
          for (let i = 0; i < normals.length; i += 3) {
            v.set(normals[i], normals[i + 1], normals[i + 2]).applyMatrix3(normalMatrix).normalize();
            normals[i] = v.x;
            normals[i + 1] = v.y;
            normals[i + 2] = v.z;
          }
        }

        draws.push({
          positions,
          normals,
          uvs: attribute(glb, primitive.attributes.TEXCOORD_0),
          colors: attribute(glb, primitive.attributes.COLOR_0),
          colorComps: primitive.attributes.COLOR_0 !== undefined
            ? { VEC3: 3, VEC4: 4 }[json.accessors[primitive.attributes.COLOR_0].type]
            : 0,
          indices: primitive.indices !== undefined ? readAccessor(glb, primitive.indices) : null,
          material: materialOf(json, primitive.material),
        });
      }
    }
    for (const child of node.children ?? []) visit(child, world);
  };
  for (const root of scene.nodes ?? []) visit(root, new Matrix4());

  if (!draws.length) throw new Error('GLB has no triangles');
  return draws;
}

function boundsOf(draws) {
  const min = new Vector3(Infinity, Infinity, Infinity);
  const max = new Vector3(-Infinity, -Infinity, -Infinity);
  for (const { positions } of draws) {
    for (let i = 0; i < positions.length; i += 3) {
      min.x = Math.min(min.x, positions[i]); max.x = Math.max(max.x, positions[i]);
      min.y = Math.min(min.y, positions[i + 1]); max.y = Math.max(max.y, positions[i + 1]);
      min.z = Math.min(min.z, positions[i + 2]); max.z = Math.max(max.z, positions[i + 2]);
    }
  }
  const center = min.clone().add(max).multiplyScalar(0.5);
  const radius = Math.max(max.clone().sub(min).length() / 2, 1e-6);
  return { min, max, center, radius };
}

/**
 * The sun's shadow map: scene depth rasterized in an orthographic frame looking along the
 * sun. Returns a sampler giving how shadowed a world position is, 0 (lit) to 1 (dark).
 */
function buildShadowMap(draws, bounds) {
  // An orthonormal basis around the sun direction, sized to cover the object.
  const zAxis = SUN.clone().negate();
  const xAxis = new Vector3(0, 1, 0).cross(zAxis).normalize();
  if (!xAxis.lengthSq()) xAxis.set(1, 0, 0);
  const yAxis = zAxis.clone().cross(xAxis).normalize();
  const span = bounds.radius * 2.1;
  const toLight = (x, y, z) => {
    const dx = x - bounds.center.x, dy = y - bounds.center.y, dz = z - bounds.center.z;
    return [
      (dx * xAxis.x + dy * xAxis.y + dz * xAxis.z) / span + 0.5,
      (dx * yAxis.x + dy * yAxis.y + dz * yAxis.z) / span + 0.5,
      (dx * zAxis.x + dy * zAxis.y + dz * zAxis.z) / span + 0.5,
    ];
  };

  // Larger light-space z is closer to the sun, so the map keeps the maximum: whatever
  // surface the sun hits first is what everything beneath it is shadowed by.
  const depth = new Float32Array(SHADOW_RES * SHADOW_RES).fill(-Infinity);
  for (const { positions, indices } of draws) {
    const count = indices ? indices.length : positions.length / 3;
    for (let t = 0; t + 2 < count; t += 3) {
      const corners = [0, 1, 2].map((k) => {
        const vi = indices ? indices[t + k] : t + k;
        return toLight(positions[vi * 3], positions[vi * 3 + 1], positions[vi * 3 + 2]);
      });
      rasterFlat(corners, SHADOW_RES, SHADOW_RES, (px, py, z) => {
        const at = py * SHADOW_RES + px;
        if (z > depth[at]) depth[at] = z;
      });
    }
  }

  return (x, y, z) => {
    const [lx, ly, lz] = toLight(x, y, z);
    const px = lx * SHADOW_RES, py = ly * SHADOW_RES;
    let hits = 0;
    for (const [ox, oy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
      const sx = Math.min(Math.max(Math.floor(px + ox - 0.5), 0), SHADOW_RES - 1);
      const sy = Math.min(Math.max(Math.floor(py + oy - 0.5), 0), SHADOW_RES - 1);
      if (lz + 0.0025 < depth[sy * SHADOW_RES + sx]) hits++;
    }
    return hits / 4;
  };
}

/** Screen-space rasterization of one triangle whose vertices are already in a unit box —
 *  used for the shadow pass, where only depth matters. */
function rasterFlat(corners, width, height, plot) {
  const [a, b, c] = corners.map(([x, y, z]) => [x * width, y * height, z]);
  const area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  if (!area) return;

  const minX = Math.max(Math.floor(Math.min(a[0], b[0], c[0])), 0);
  const maxX = Math.min(Math.ceil(Math.max(a[0], b[0], c[0])), width - 1);
  const minY = Math.max(Math.floor(Math.min(a[1], b[1], c[1])), 0);
  const maxY = Math.min(Math.ceil(Math.max(a[1], b[1], c[1])), height - 1);

  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const px = x + 0.5, py = y + 0.5;
      const w0 = ((b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])) / area;
      const w1 = ((c[0] - b[0]) * (py - b[1]) - (c[1] - b[1]) * (px - b[0])) / area;
      const w2 = 1 - w0 - w1;
      // Sign-agnostic inside test: shadow casting has no facing.
      if ((w0 < 0 || w1 < 0 || w2 < 0) && (w0 > 0 || w1 > 0 || w2 > 0)) continue;
      plot(x, y, w1 * a[2] + w2 * b[2] + w0 * c[2]);
    }
  }
}

/**
 * How far the object's projected silhouette sits from the middle of the frame, in NDC.
 *
 * Aiming the camera at the bounding-box centre does not centre the pixels: perspective makes
 * the nearer side project larger, so the silhouette drifts off-middle. Offsetting the
 * projection by this lands the silhouette's own bounds dead centre. It cannot push anything
 * out of frame either — the recentred half-extent `(max - min) / 2` is never larger than the
 * `max(|min|, |max|)` it replaces.
 */
function ndcOffset(draws, viewProjection) {
  const e = viewProjection.elements;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const { positions } of draws) {
    for (let i = 0; i < positions.length; i += 3) {
      const x = positions[i], y = positions[i + 1], z = positions[i + 2];
      const w = e[3] * x + e[7] * y + e[11] * z + e[15];
      if (w <= 1e-6) continue;
      const nx = (e[0] * x + e[4] * y + e[8] * z + e[12]) / w;
      const ny = (e[1] * x + e[5] * y + e[9] * z + e[13]) / w;
      if (nx < minX) minX = nx;
      if (nx > maxX) maxX = nx;
      if (ny < minY) minY = ny;
      if (ny > maxY) maxY = ny;
    }
  }
  return minX > maxX ? [0, 0] : [-(minX + maxX) / 2, -(minY + maxY) / 2];
}

/**
 * Every GLB's draws, with each one's textures resolved onto the draw itself.
 *
 * Resolving here rather than at shade time is what lets several GLBs share one frame:
 * `material.image` indexes that GLB's own image list, so two files each mean something
 * different by image 0 and merging their draws naively swaps their textures.
 */
async function prepare(glbs) {
  const all = [];
  for (const [objectIndex, glb] of glbs.entries()) {
    const draws = gatherDraws(glb);
    const images = await decodeImages(glb, draws);
    for (const draw of draws) {
      draw.texture = draw.material.image !== null ? (images.get(draw.material.image) ?? null) : null;
      draw.objectIndex = objectIndex;
      all.push(draw);
    }
  }
  return all;
}

/**
 * One camera over an already-prepared scene, out to a PNG buffer.
 *
 * `direction` points from the subject towards the eye and needs an `up` that is not parallel
 * to it — which is why the top and bottom views carry their own.
 */
function renderPass(draws, bounds, shadowAt, { direction, up, size, exposure, floor, clip = null, palette = null }) {
  // Maps a shaded channel onto [floor, 1] and out to bytes.
  const span = 255 * (1 - floor);
  const base = 255 * floor;
  const lift = (v) => base + span * (v > 1 ? 1 : v > 0 ? v : 0);

  // Camera: far enough back along `direction` that the bounding sphere fits.
  const distance = (bounds.radius / Math.sin(FOV / 2)) * 1.12;
  const eye = direction.clone().normalize().multiplyScalar(distance).add(bounds.center);
  // three's lookAt fills in rotation only, so the eye has to be set before inverting.
  const view = new Matrix4().lookAt(eye, bounds.center, up).setPosition(eye).invert();
  const near = Math.max(distance - bounds.radius * 2.5, distance * 0.05);
  const far = distance + bounds.radius * 2.5;
  const projection = new Matrix4().makePerspective(
    -near * Math.tan(FOV / 2), near * Math.tan(FOV / 2),
    near * Math.tan(FOV / 2), -near * Math.tan(FOV / 2),
    near, far,
  );
  const viewProjection = projection.clone().multiply(view);
  const [offsetX, offsetY] = ndcOffset(draws, viewProjection);

  const res = size * 2;
  // Opaque black from the start, so the downsample blends object edges against the
  // background rather than against nothing.
  const color = new Uint8ClampedArray(res * res * 4);
  for (let at = 3; at < color.length; at += 4) color[at] = 255;
  const zbuffer = new Float32Array(res * res).fill(Infinity);

  // Projects a world position to [screenX, screenY, ndcDepth, 1/w] — null when behind us.
  const project = (x, y, z) => {
    const cx = viewProjection.elements[0] * x + viewProjection.elements[4] * y + viewProjection.elements[8] * z + viewProjection.elements[12];
    const cy = viewProjection.elements[1] * x + viewProjection.elements[5] * y + viewProjection.elements[9] * z + viewProjection.elements[13];
    const cz = viewProjection.elements[2] * x + viewProjection.elements[6] * y + viewProjection.elements[10] * z + viewProjection.elements[14];
    const cw = viewProjection.elements[3] * x + viewProjection.elements[7] * y + viewProjection.elements[11] * z + viewProjection.elements[15];
    if (cw <= 1e-6) return null;
    return [
      (cx / cw + offsetX + 1) * 0.5 * res,
      (1 - (cy / cw + offsetY)) * 0.5 * res,
      cz / cw,
      1 / cw,
    ];
  };

  // --- the object ---------------------------------------------------------------------
  const viewDir = new Vector3();
  for (const draw of draws) {
    const { positions, normals, uvs, colors, colorComps, indices, material, texture } = draw;
    const count = indices ? indices.length : positions.length / 3;
    // A palette replaces the material's own colour, so the texture is only worth sampling
    // where its alpha still decides whether the fragment exists at all.
    const tint = palette?.[draw.objectIndex] ?? null;
    const sampleTexture = texture && uvs && (!tint || material.alphaCutoff !== null);

    for (let t = 0; t + 2 < count; t += 3) {
      const verts = [];
      let behind = false;
      for (let k = 0; k < 3; k++) {
        const vi = indices ? indices[t + k] : t + k;
        const world = [positions[vi * 3], positions[vi * 3 + 1], positions[vi * 3 + 2]];
        const screen = project(world[0], world[1], world[2]);
        if (!screen) { behind = true; break; }
        verts.push({ vi, world, screen });
      }
      if (behind) continue;

      // Flat face normal fallback, and the two-sided flip so backfaces still shade.
      let fnx = 0, fny = 0, fnz = 0;
      if (!normals) {
        const [p, q, r] = verts.map((v) => v.world);
        fnx = (q[1] - p[1]) * (r[2] - p[2]) - (q[2] - p[2]) * (r[1] - p[1]);
        fny = (q[2] - p[2]) * (r[0] - p[0]) - (q[0] - p[0]) * (r[2] - p[2]);
        fnz = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
        const len = Math.hypot(fnx, fny, fnz) || 1;
        fnx /= len; fny /= len; fnz /= len;
      }

      rasterTriangle(verts, res, zbuffer, (at, world, bary) => {
        if (clip?.objectIndex === draw.objectIndex && clip.reject(world)) return false;

        // Attributes, perspective-corrected by the rasterizer's barycentrics.
        let r = material.factor[0], g = material.factor[1], b = material.factor[2];
        let textureAlpha = 1;

        if (sampleTexture) {
          let u = 0, v = 0;
          for (let k = 0; k < 3; k++) {
            u += bary[k] * uvs[verts[k].vi * 2];
            v += bary[k] * uvs[verts[k].vi * 2 + 1];
          }
          u = u * material.uvScale[0] + material.uvOffset[0];
          v = v * material.uvScale[1] + material.uvOffset[1];
          const tx = Math.min(Math.max(Math.floor((u - Math.floor(u)) * texture.width), 0), texture.width - 1);
          const ty = Math.min(Math.max(Math.floor((v - Math.floor(v)) * texture.height), 0), texture.height - 1);
          const ti = (ty * texture.width + tx) * 4;
          r *= texture.data[ti] / 255;
          g *= texture.data[ti + 1] / 255;
          b *= texture.data[ti + 2] / 255;
          textureAlpha = texture.data[ti + 3] / 255;
        }
        if (colors && !tint) {
          let cr = 0, cg = 0, cb = 0;
          for (let k = 0; k < 3; k++) {
            cr += bary[k] * colors[verts[k].vi * colorComps];
            cg += bary[k] * colors[verts[k].vi * colorComps + 1];
            cb += bary[k] * colors[verts[k].vi * colorComps + 2];
          }
          r *= cr; g *= cg; b *= cb;
        }
        // Only a texture's alpha carves a shape. A material-wide alpha is glass, and a glass
        // bowl held to the same 0.5 test loses every fragment it has: the object disappears
        // rather than being seen through, which is indistinguishable from never being placed.
        if (material.alphaCutoff !== null && textureAlpha < material.alphaCutoff) return false;
        // After the cutout test, so a leaf card keeps its shape and loses only its colour.
        if (tint) { r = tint[0]; g = tint[1]; b = tint[2]; }

        let nx = fnx, ny = fny, nz = fnz;
        if (normals) {
          nx = ny = nz = 0;
          for (let k = 0; k < 3; k++) {
            nx += bary[k] * normals[verts[k].vi * 3];
            ny += bary[k] * normals[verts[k].vi * 3 + 1];
            nz += bary[k] * normals[verts[k].vi * 3 + 2];
          }
        }
        viewDir.set(eye.x - world[0], eye.y - world[1], eye.z - world[2]);
        if (nx * viewDir.x + ny * viewDir.y + nz * viewDir.z < 0) { nx = -nx; ny = -ny; nz = -nz; }
        const nlen = Math.hypot(nx, ny, nz) || 1;

        const diffuse = Math.max(-(nx * SUN.x + ny * SUN.y + nz * SUN.z) / nlen, 0);
        const sky = 0.5 + (0.5 * ny) / nlen;
        const lit = 1 - SHADOW_DEPTH * shadowAt(world[0], world[1], world[2]);
        // `viewDir` already points at the eye and is not unit length.
        const vlen = Math.hypot(viewDir.x, viewDir.y, viewDir.z) || 1;
        const facing = Math.max((nx * viewDir.x + ny * viewDir.y + nz * viewDir.z) / (nlen * vlen), 0);
        const shade =
          (AMBIENT * (0.65 + 0.35 * sky) + SUN_STRENGTH * diffuse * lit + FILL * facing) * exposure;

        color[at * 4] = lift(r * shade);
        color[at * 4 + 1] = lift(g * shade);
        color[at * 4 + 2] = lift(b * shade);
        return true;
      });
    }
  }

  return sharp(Buffer.from(color.buffer), { raw: { width: res, height: res, channels: 4 } })
    .resize(size, size, { kernel: 'lanczos3' })
    // Three channels out: the background is black, not absent, and an alpha channel that is
    // 255 everywhere only invites something downstream to composite against it.
    .removeAlpha()
    .png()
    .toBuffer();
}

/** Renders a parsed GLB to a PNG buffer, from its glTF front with the exporter's pitch. */
export async function renderGLB(glb, { size = SIZE(), exposure = EXPOSURE(), floor = BLACK_FLOOR() } = {}) {
  const draws = await prepare([glb]);
  const bounds = boundsOf(draws);
  return renderPass(draws, bounds, buildShadowMap(draws, bounds), {
    direction: new Vector3(0, Math.sin(PITCH), Math.cos(PITCH)),
    up: new Vector3(0, 1, 0),
    size,
    exposure,
    floor,
  });
}

/**
 * Six exterior review views: four cardinal cameras elevated 15° above the scene centre,
 * then unchanged axial top and bottom cameras. The anchor cutaway is added separately.
 */
const CARDINAL_ELEVATION = (15 * Math.PI) / 180;
const CARDINAL_HORIZONTAL = Math.cos(CARDINAL_ELEVATION);
const CARDINAL_VERTICAL = Math.sin(CARDINAL_ELEVATION);

export const VIEWS = [
  { name: 'front', direction: [0, CARDINAL_VERTICAL, CARDINAL_HORIZONTAL], up: [0, 1, 0] },
  { name: 'right', direction: [CARDINAL_HORIZONTAL, CARDINAL_VERTICAL, 0], up: [0, 1, 0] },
  { name: 'back', direction: [0, CARDINAL_VERTICAL, -CARDINAL_HORIZONTAL], up: [0, 1, 0] },
  { name: 'left', direction: [-CARDINAL_HORIZONTAL, CARDINAL_VERTICAL, 0], up: [0, 1, 0] },
  { name: 'top', direction: [0, 1, 0], up: [0, 0, -1] },
  { name: 'bottom', direction: [0, -1, 0], up: [0, 0, 1] },
];

/**
 * Untextured review colours: a neutral anchor and a red placed object.
 *
 * Texture is what a reader has to see past to answer the only question being asked — which
 * of the two objects is which, and where one sits against the other. Two flat albedos leave
 * the lighting to carry the form and make the boundary between them unambiguous even where
 * the meshes touch, interpenetrate or share a palette.
 */
export const REVIEW_PALETTE = [
  [0.74, 0.75, 0.78],
  [0.92, 0.09, 0.09],
];

/**
 * Several posed GLBs in one frame, rendered from each of `views`.
 *
 * The meshes carry their own placement, so nothing is transformed here — what the images show
 * is what the files say, which is the whole point of using them to check the files. Bounds and
 * the shadow map are computed once over the union: framing every view identically is what
 * makes the exterior views and cutaway comparable, and the shadow map depends only on
 * geometry and the sun.
 */
export async function renderViews(
  glbs,
  { size = SIZE(), exposure = EXPOSURE(), floor = BLACK_FLOOR(), views = VIEWS, palette = REVIEW_PALETTE } = {},
) {
  const draws = await prepare(glbs);
  const bounds = boundsOf(draws);
  const shadowAt = buildShadowMap(draws, bounds);

  const out = [];
  for (const view of views) {
    out.push({
      name: view.name,
      png: await renderPass(draws, bounds, shadowAt, {
        direction: new Vector3().fromArray(view.direction),
        up: new Vector3().fromArray(view.up),
        size,
        exposure,
        floor,
        palette,
      }),
    });
  }

  if (glbs.length > 1) {
    const direction = new Vector3().fromArray(VIEWS[0].direction).normalize();
    const anchorCenter = boundsOf(draws.filter((draw) => draw.objectIndex === 0)).center;
    out.push({
      name: 'anchor-cutaway',
      png: await renderPass(draws, bounds, shadowAt, {
        direction,
        up: new Vector3().fromArray(VIEWS[0].up),
        size,
        exposure,
        floor,
        palette,
        // Remove the camera-facing half of A. B stays intact, exposing containment while
        // preserving the same front framing as image 1.
        clip: {
          objectIndex: 0,
          reject: ([x, y, z]) =>
            (x - anchorCenter.x) * direction.x +
              (y - anchorCenter.y) * direction.y +
              (z - anchorCenter.z) * direction.z >
            0,
        },
      }),
    });
  }
  return out;
}

/**
 * Perspective-correct triangle rasterization against the shared z-buffer. `plot` gets the
 * pixel index, the interpolated world position and perspective-corrected barycentrics; it
 * returns false to reject the fragment (alpha cutout) without writing depth.
 */
function rasterTriangle(verts, res, zbuffer, plot) {
  const [a, b, c] = verts.map((v) => v.screen);
  const area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  if (!area) return;

  const minX = Math.max(Math.floor(Math.min(a[0], b[0], c[0])), 0);
  const maxX = Math.min(Math.ceil(Math.max(a[0], b[0], c[0])), res - 1);
  const minY = Math.max(Math.floor(Math.min(a[1], b[1], c[1])), 0);
  const maxY = Math.min(Math.ceil(Math.max(a[1], b[1], c[1])), res - 1);

  const bary = [0, 0, 0];
  const world = [0, 0, 0];
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const px = x + 0.5, py = y + 0.5;
      const w0 = ((b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])) / area;
      const w1 = ((c[0] - b[0]) * (py - b[1]) - (c[1] - b[1]) * (px - b[0])) / area;
      const w2 = 1 - w0 - w1;
      if ((w0 < 0 || w1 < 0 || w2 < 0) && (w0 > 0 || w1 > 0 || w2 > 0)) continue;

      // Affine weights (screen space) for depth; divide by w for world/attributes.
      const depth = w1 * a[2] + w2 * b[2] + w0 * c[2];
      const at = y * res + x;
      if (depth >= zbuffer[at]) continue;

      const iw = w1 * a[3] + w2 * b[3] + w0 * c[3];
      bary[0] = (w1 * a[3]) / iw;
      bary[1] = (w2 * b[3]) / iw;
      bary[2] = (w0 * c[3]) / iw;
      for (let k = 0; k < 3; k++) {
        world[k] = bary[0] * verts[0].world[k] + bary[1] * verts[1].world[k] + bary[2] * verts[2].world[k];
      }
      if (plot(at, world, bary) !== false) zbuffer[at] = depth;
    }
  }
}
