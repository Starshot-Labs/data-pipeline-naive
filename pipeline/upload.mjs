// Step 7: finish publishing a sample.
//
//   node pipeline/upload.mjs           publish everything placed that is not up yet
//   node pipeline/upload.mjs --force   re-publish everything placed
//
// Almost nothing happens here any more. Baking already wrote the posed meshes and their
// reference images into `<prefix>/<id>/` on the scene volume, straight from one volume to
// another, so all that is left is the pair of small text files this machine owns:
// metadata.json and placement.txt.
//
// They go last on purpose. A folder with meshes but no metadata.json is a sample mid-flight;
// one with metadata.json is finished, which is exactly the test `published` applies.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { mapLimit, retry, widthOf } from './limit.mjs';
import * as scene from './scene.mjs';
import * as meta from './metadata.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ENV_FILE = path.join(ROOT, '.env');
if (fs.existsSync(ENV_FILE)) process.loadEnvFile(ENV_FILE);

const GENERATED_DIR = path.resolve(ROOT, process.env.GENERATED_DIR ?? 'generated');
// Lower than the model stages by default: this one answers to `dc-scene-ops`, which
// serialises volume writes per container, so extra width only deepens a queue.
const PUBLISH_WIDTH = widthOf('PUBLISH_CONCURRENCY', 50);
// Samples per call. A sample's two files are a couple of kilobytes, so the batch exists to
// amortise the round trip and the volume commit, both of which cost most of a second.
const PUBLISH_BATCH = Number(process.env.PUBLISH_BATCH ?? 250);

const payloadOf = (sample) => ({
  sample: sample.id,
  files: {
    'metadata.json': `${JSON.stringify(sample.metadata, null, 2)}\n`,
    'placement.txt': sample.metadata.placement,
  },
});

/** Publishes every placed sample not already up; returns how many went. */
export async function uploadSamples({ force = false } = {}) {
  const placed = meta.list(GENERATED_DIR).filter((sample) => meta.isPlaced(sample.metadata));
  if (!placed.length) {
    console.log('  nothing placed to publish');
    return 0;
  }

  const already = force ? new Set() : await scene.published();
  const pending = placed.filter((sample) => !already.has(sample.id));
  if (!pending.length) {
    console.log(`  all ${placed.length} placed sample(s) are already up`);
    return 0;
  }

  const batches = [];
  for (let i = 0; i < pending.length; i += PUBLISH_BATCH) batches.push(pending.slice(i, i + PUBLISH_BATCH));

  // A batch fails or lands together, which is the trade for one commit instead of hundreds.
  // Nothing is lost either way: an unpublished sample keeps its metadata and goes next run.
  const results = await mapLimit(batches, PUBLISH_WIDTH, async (batch) => {
    try {
      const { published } = await retry(() => scene.publish(batch.map(payloadOf)));
      return published.length;
    } catch (err) {
      console.error(`    ✗ ${batch.length} sample(s) from ${batch[0].id}: ${err.message}`);
      return 0;
    }
  });

  const done = results.reduce((total, n) => total + n, 0);
  console.log(`  ✓ ${done}/${pending.length} sample(s) published in ${batches.length} call(s)`);
  return done;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  console.log(`\n7 · publishing via ${scene.url()}`);
  try {
    await uploadSamples({ force: process.argv.includes('--force') });
  } catch (err) {
    console.error(`  ✗ ${err.message}`);
    process.exitCode = 1;
  }
}
