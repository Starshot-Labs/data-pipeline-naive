// An isometric shot of a mesh, rasterized in-process so nothing needs a GPU or a
// browser: orthographic projection, z-buffered edge-function fill, flat Lambert
// shading on a white backdrop. Deliberately material-free — its consumers are an
// image model that needs to see the anchor's form, and a results folder that
// wants a quick visual check of a placement.

import sharp from 'sharp';
import { Vector3, Matrix4, MathUtils } from 'three';

// The classic isometric camera: 45° around, atan(1/sqrt(2)) up.
export const ISO_VIEW = { azimuth_deg: 45, elevation_deg: 35.264 };

const MARGIN = 0.9; // fraction of the frame the mesh's longest screen axis fills
const SUPERSAMPLE = 2; // rendered at 2x and averaged down, an economical antialias
const AMBIENT = 0.35;
const DIFFUSE = 0.65;
const BASE = 235; // the shade of a face square to the light
const LIGHT = new Vector3(-0.35, 0.75, 0.6).normalize(); // view space, over the left shoulder

/** World → view matrix for a camera on the unit sphere looking at the origin. */
function viewMatrix({ azimuth_deg, elevation_deg }) {
  const azimuth = MathUtils.degToRad(azimuth_deg);
  const elevation = MathUtils.degToRad(elevation_deg);
  const eye = new Vector3(
    Math.sin(azimuth) * Math.cos(elevation),
    Math.sin(elevation),
    Math.cos(azimuth) * Math.cos(elevation),
  );
  return new Matrix4().lookAt(eye, new Vector3(0, 0, 0), new Vector3(0, 1, 0)).setPosition(eye).invert();
}

/**
 * PNG bytes of one or more triangle sets seen from `view`, framed together.
 * Each layer is `{ triangles, base?, dither? }` — `triangles` flat 9-float
 * world-space triangles as produced by `sceneTriangles`, `base` the colour of a
 * face square to the light as one number for a grey or `[r, g, b]`, and `dither`
 * renders it screen-door transparent (every other subpixel), which the
 * supersampled downscale turns into a smooth ghost. That is what lets a scene
 * view show an object placed inside a cavity. The projection is orthographic,
 * so only the view direction matters, never a camera distance.
 */
export async function renderView(layers, { size = 1024, view = ISO_VIEW } = {}) {
  const matrix = viewMatrix(view);
  const canvas = size * SUPERSAMPLE;

  // One pass into view space, so the shared fit and the fill read the same coordinates.
  const vertex = new Vector3();
  let [minX, minY, maxX, maxY] = [Infinity, Infinity, -Infinity, -Infinity];
  const projectedLayers = layers.map(({ triangles, base = BASE, dither = false }) => {
    const tint = Array.isArray(base) ? base : [base, base, base];
    const projected = new Float64Array(triangles.length);
    for (let i = 0; i < triangles.length; i += 3) {
      vertex.set(triangles[i], triangles[i + 1], triangles[i + 2]).applyMatrix4(matrix);
      projected[i] = vertex.x;
      projected[i + 1] = vertex.y;
      projected[i + 2] = vertex.z;
      if (vertex.x < minX) minX = vertex.x;
      if (vertex.x > maxX) maxX = vertex.x;
      if (vertex.y < minY) minY = vertex.y;
      if (vertex.y > maxY) maxY = vertex.y;
    }
    return { projected, tint, dither };
  });

  const scale = (canvas * MARGIN) / Math.max(maxX - minX, maxY - minY);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const toScreenX = (x) => (x - centerX) * scale + canvas / 2;
  const toScreenY = (y) => canvas / 2 - (y - centerY) * scale; // +Y up becomes row 0 at the top

  const pixels = new Uint8Array(canvas * canvas * 3).fill(255);
  const depth = new Float64Array(canvas * canvas).fill(-Infinity); // view-space z, nearer is larger

  for (const { projected, tint, dither } of projectedLayers) {
    for (let t = 0; t < projected.length; t += 9) {
      const x0 = toScreenX(projected[t]), y0 = toScreenY(projected[t + 1]), z0 = projected[t + 2];
      const x1 = toScreenX(projected[t + 3]), y1 = toScreenY(projected[t + 4]), z1 = projected[t + 5];
      const x2 = toScreenX(projected[t + 6]), y2 = toScreenY(projected[t + 7]), z2 = projected[t + 8];

      const area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
      if (Math.abs(area) < 1e-9) continue;

      // Flat shade off the view-space normal, flipped toward the camera so open
      // meshes read the same from either side.
      const ux = projected[t + 3] - projected[t], uy = projected[t + 4] - projected[t + 1], uz = projected[t + 5] - projected[t + 2];
      const vx = projected[t + 6] - projected[t], vy = projected[t + 7] - projected[t + 1], vz = projected[t + 8] - projected[t + 2];
      let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      const length = Math.hypot(nx, ny, nz);
      if (length === 0) continue;
      if (nz < 0) { nx = -nx; ny = -ny; nz = -nz; }
      const lambert = Math.max(0, (nx * LIGHT.x + ny * LIGHT.y + nz * LIGHT.z) / length);
      const lit = AMBIENT + DIFFUSE * lambert;
      const red = Math.min(255, Math.round(tint[0] * lit));
      const green = Math.min(255, Math.round(tint[1] * lit));
      const blue = Math.min(255, Math.round(tint[2] * lit));

      const left = Math.max(0, Math.floor(Math.min(x0, x1, x2)));
      const right = Math.min(canvas - 1, Math.ceil(Math.max(x0, x1, x2)));
      const top = Math.max(0, Math.floor(Math.min(y0, y1, y2)));
      const bottom = Math.min(canvas - 1, Math.ceil(Math.max(y0, y1, y2)));

      for (let py = top; py <= bottom; py++) {
        for (let px = left; px <= right; px++) {
          if (dither && ((px + py) & 1)) continue;
          const cx = px + 0.5, cy = py + 0.5;
          const w0 = (x1 - cx) * (y2 - cy) - (y1 - cy) * (x2 - cx);
          const w1 = (x2 - cx) * (y0 - cy) - (y2 - cy) * (x0 - cx);
          const w2 = (x0 - cx) * (y1 - cy) - (y0 - cy) * (x1 - cx);
          const inside = (w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0);
          if (!inside) continue;

          const z = (w0 * z0 + w1 * z1 + w2 * z2) / area;
          const at = py * canvas + px;
          if (z <= depth[at]) continue;
          depth[at] = z;
          const o = at * 3;
          pixels[o] = red;
          pixels[o + 1] = green;
          pixels[o + 2] = blue;
        }
      }
    }
  }

  return sharp(pixels, { raw: { width: canvas, height: canvas, channels: 3 } })
    .resize(size, size)
    .png()
    .toBuffer();
}
