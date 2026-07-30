// A software renderer for the per-object reference images: one GLB in, one PNG out.
//
// Pure JS on purpose. Blender or a GPU context would render prettier, but this has to run
// wherever the pipeline runs — a Modal CPU container, a laptop, inside dc-scene-ops — with
// zero native or system dependencies beyond what the pipeline already carries (`three` for
// the math, `sharp` to decode textures and encode the PNG). "Basic rendering" is the spec:
// textured, sun-lit, shadowed, and fast; not path-traced.
//
// The look: the object centred against a transparent background, seen from its glTF front
// (+Z) with the same 10° downward pitch the manual exporter uses, lit by one directional
// sun plus a sky ambient, with a shadow map for self-shadowing and a soft catcher disc on
// the ground plane so the object reads as grounded rather than pasted.
//
// Rendered at 2× and downsampled, which is the cheapest anti-aliasing there is.

import sharp from 'sharp';
import { Matrix3, Matrix4, Vector3 } from 'three';
import { readAccessor, localMatrix } from './glb.mjs';

const SIZE = () => Number(process.env.RENDER_SIZE ?? 512);
const FOV = (35 * Math.PI) / 180;
const PITCH = (10 * Math.PI) / 180;
const SHADOW_RES = 1024;
// Down-and-inward from the upper front-left, so form shows on the front faces the camera sees.
const SUN = new Vector3(-0.45, -1, -0.35).normalize();

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
    // BLEND is treated as a 0.5 cutout: correct sorting is not worth its cost here.
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

const smoothstep = (lo, hi, x) => {
  const t = Math.min(Math.max((x - lo) / (hi - lo), 0), 1);
  return t * t * (3 - 2 * t);
};

/**
 * The sun's shadow map: scene depth rasterized in an orthographic frame looking along the
 * sun. Returns a sampler giving how shadowed a world position is, 0 (lit) to 1 (dark).
 */
function buildShadowMap(draws, bounds, groundExtent) {
  // An orthonormal basis around the sun direction, sized to cover object and catcher disc.
  const zAxis = SUN.clone().negate();
  const xAxis = new Vector3(0, 1, 0).cross(zAxis).normalize();
  if (!xAxis.lengthSq()) xAxis.set(1, 0, 0);
  const yAxis = zAxis.clone().cross(xAxis).normalize();
  const span = Math.max(bounds.radius, groundExtent) * 2.1;
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

/** Renders a parsed GLB to a PNG buffer. */
export async function renderGLB(glb, { size = SIZE() } = {}) {
  const draws = gatherDraws(glb);
  const images = await decodeImages(glb, draws);
  const bounds = boundsOf(draws);

  const groundY = bounds.min.y;
  const groundExtent = bounds.radius * 1.8;
  const shadowAt = buildShadowMap(draws, bounds, groundExtent);

  // Camera: on the +Z side, pitched down, far enough back that the bounding sphere fits.
  const distance = (bounds.radius / Math.sin(FOV / 2)) * 1.12;
  const eye = new Vector3(0, Math.sin(PITCH), Math.cos(PITCH)).multiplyScalar(distance).add(bounds.center);
  // three's lookAt fills in rotation only, so the eye has to be set before inverting.
  const view = new Matrix4().lookAt(eye, bounds.center, new Vector3(0, 1, 0)).setPosition(eye).invert();
  const near = Math.max(distance - bounds.radius * 2.5, distance * 0.05);
  const far = distance + bounds.radius * 2.5;
  const projection = new Matrix4().makePerspective(
    -near * Math.tan(FOV / 2), near * Math.tan(FOV / 2),
    near * Math.tan(FOV / 2), -near * Math.tan(FOV / 2),
    near, far,
  );
  const viewProjection = projection.clone().multiply(view);

  const res = size * 2;
  const color = new Uint8ClampedArray(res * res * 4);
  const zbuffer = new Float32Array(res * res).fill(Infinity);

  // Projects a world position to [screenX, screenY, ndcDepth, 1/w] — null when behind us.
  const project = (x, y, z) => {
    const cx = viewProjection.elements[0] * x + viewProjection.elements[4] * y + viewProjection.elements[8] * z + viewProjection.elements[12];
    const cy = viewProjection.elements[1] * x + viewProjection.elements[5] * y + viewProjection.elements[9] * z + viewProjection.elements[13];
    const cz = viewProjection.elements[2] * x + viewProjection.elements[6] * y + viewProjection.elements[10] * z + viewProjection.elements[14];
    const cw = viewProjection.elements[3] * x + viewProjection.elements[7] * y + viewProjection.elements[11] * z + viewProjection.elements[15];
    if (cw <= 1e-6) return null;
    return [((cx / cw) + 1) * 0.5 * res, (1 - (cy / cw)) * 0.5 * res, cz / cw, 1 / cw];
  };

  // --- ground catcher: a soft dark disc where the sun is blocked, alpha-only ---------
  {
    const e = groundExtent;
    const cx = bounds.center.x, cz = bounds.center.z;
    const quad = [
      [cx - e, groundY, cz - e], [cx + e, groundY, cz - e], [cx + e, groundY, cz + e],
      [cx - e, groundY, cz - e], [cx + e, groundY, cz + e], [cx - e, groundY, cz + e],
    ];
    for (let t = 0; t < 6; t += 3) {
      rasterTriangle(quad.slice(t, t + 3).map(([x, y, z]) => ({ world: [x, y, z], screen: project(x, y, z) })), res, zbuffer, (at, world) => {
        const radial = Math.hypot(world[0] - cx, world[2] - cz) / e;
        if (radial > 1) return;
        const darkness = shadowAt(world[0], world[1], world[2]) * 0.4 * (1 - smoothstep(0.45, 1, radial));
        if (darkness <= 0.01) return;
        color[at * 4 + 3] = Math.max(color[at * 4 + 3], darkness * 255);
      });
    }
  }

  // --- the object ---------------------------------------------------------------------
  const viewDir = new Vector3();
  for (const draw of draws) {
    const { positions, normals, uvs, colors, colorComps, indices, material } = draw;
    const texture = material.image !== null ? images.get(material.image) : undefined;
    const count = indices ? indices.length : positions.length / 3;

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
        // Attributes, perspective-corrected by the rasterizer's barycentrics.
        let r = material.factor[0], g = material.factor[1], b = material.factor[2], a = material.factor[3];

        if (texture && uvs) {
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
          a *= texture.data[ti + 3] / 255;
        }
        if (colors) {
          let cr = 0, cg = 0, cb = 0;
          for (let k = 0; k < 3; k++) {
            cr += bary[k] * colors[verts[k].vi * colorComps];
            cg += bary[k] * colors[verts[k].vi * colorComps + 1];
            cb += bary[k] * colors[verts[k].vi * colorComps + 2];
          }
          r *= cr; g *= cg; b *= cb;
        }
        if (material.alphaCutoff !== null && a < material.alphaCutoff) return false;

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
        const lit = 1 - 0.85 * shadowAt(world[0], world[1], world[2]);
        const shade = 0.32 * (0.65 + 0.35 * sky) + 0.78 * diffuse * lit;

        color[at * 4] = r * shade * 255;
        color[at * 4 + 1] = g * shade * 255;
        color[at * 4 + 2] = b * shade * 255;
        color[at * 4 + 3] = 255;
        return true;
      });
    }
  }

  return sharp(Buffer.from(color.buffer), { raw: { width: res, height: res, channels: 4 } })
    .resize(size, size, { kernel: 'lanczos3' })
    .png()
    .toBuffer();
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
