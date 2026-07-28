// Stages 1-2 on their own: invent samples and render their reference images.
//
// Worth running apart from the mesh half because the two have nothing in common
// operationally. This half is cheap, needs no GPU, and runs 50 requests wide, so a corpus
// can be prepared whenever — including while the farm is busy with someone else's campaign.
//
//   node pipeline/build-images.mjs --pairs=50   invent 50 pairs, then render what is missing
//   node pipeline/build-images.mjs              only fill in images that failed earlier

import { buildImages } from './build.mjs';

const pairs = Number(process.argv.slice(2).find((arg) => arg.startsWith('--pairs='))?.slice(8) ?? 0);
if (!Number.isInteger(pairs) || pairs < 0) {
  console.error('usage: node pipeline/build-images.mjs [--pairs=N]');
  process.exit(2);
}

const ready = await buildImages({ pairs });
if (ready.length) console.log('Mesh them with:  node pipeline/build-meshes.mjs');
