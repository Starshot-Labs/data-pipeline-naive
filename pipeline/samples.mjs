// A placement-set sample, which is the input both experiments read: one anchor GLB, one
// photo of the object to place, and the placement phrase on its own line in placement.txt.

import fs from 'node:fs';
import path from 'node:path';
import { IMAGE_EXT } from './metadata.mjs';

export const listSamples = (setDir) =>
  fs.readdirSync(setDir, { withFileTypes: true }).filter((entry) => entry.isDirectory()).map((entry) => entry.name);

export function readSample(setDir, id) {
  const dir = path.join(setDir, id);
  if (!fs.existsSync(dir)) throw new Error('no such sample folder');
  const files = fs.readdirSync(dir);

  const meshes = files.filter((file) => file.toLowerCase().endsWith('.glb'));
  const images = files.filter((file) => IMAGE_EXT.test(file));
  if (meshes.length !== 1) throw new Error(`expected exactly one anchor .glb, found ${meshes.length}`);
  if (images.length !== 1) throw new Error(`expected exactly one source image, found ${images.length}`);

  const phrase = fs.existsSync(path.join(dir, 'placement.txt'))
    ? fs.readFileSync(path.join(dir, 'placement.txt'), 'utf8').trim()
    : '';
  if (!phrase) throw new Error('placement.txt is missing or empty');

  return { dir, anchorFile: meshes[0], imageFile: images[0], phrase };
}

/** "cook_dining_chair_end_right.glb" → "cook dining chair end right", for a model to read. */
export const nameOf = (file) =>
  path.basename(file).replace(/\.(glb|png|jpe?g|webp)$/i, '').replace(/\.raw$/i, '').replace(/[_-]+/g, ' ').trim();
