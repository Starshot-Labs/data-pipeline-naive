// The whole pipeline, stages 1-7, in one command.
//
//   node pipeline/all.mjs 50   invent 50 object pairs and carry them through to a published dataset
//   node pipeline/all.mjs      carry whatever is already on disk as far as it will go
//
// Safe to re-run: each stage skips work it already finished, so a second pass only fills the
// gaps left by a failed image, mesh, placement or upload.

import path from 'node:path';
import { buildSamples, GENERATED_DIR } from './build.mjs';
import { placeSamples } from './run.mjs';
import { uploadSamples } from './upload.mjs';
import * as scene from './scene.mjs';
import * as meta from './metadata.mjs';

const pairs = Number(process.argv.slice(2).find((arg) => !arg.startsWith('--')) ?? 0);
if (!Number.isInteger(pairs) || pairs < 0) {
  console.error('usage: node pipeline/all.mjs [pairs]');
  process.exit(2);
}

const ready = await buildSamples({ pairs });
const pending = ready.filter((id) => !meta.isPlaced(meta.read(path.join(GENERATED_DIR, id))));

let failed = 0;
console.log(`\n4-6 · placing ${pending.length} sample(s)`);
if (!pending.length) {
  console.log(ready.length ? '  everything with meshes is already placed' : '  nothing has meshes yet');
} else {
  failed = await placeSamples({ sourceDir: GENERATED_DIR, ids: pending });
}

console.log(`\n7 · publishing via ${scene.url()}`);
try {
  await uploadSamples();
} catch (err) {
  console.error(`  ✗ ${err.message}`);
  failed++;
}

if (failed) process.exitCode = 1;
