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
import { track, timed } from './progress.mjs';
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

// Serialised through `meta` rather than here, so the published file is byte-for-byte the one
// sitting in the sample folder — same key order, same trailing newline.
const payloadOf = (sample) => ({
  sample: sample.id,
  files: {
    'metadata.json': meta.serialize(sample.metadata),
    'placement.txt': meta.placementText(sample.metadata),
  },
});

/**
 * The placed samples of a corpus, read concurrently.
 *
 * `meta.list` reads each metadata.json with a synchronous `readFileSync`, which is fine for
 * hundreds and ruinous for tens of thousands: none of them are in a fresh container's cache,
 * so it becomes that many serial round trips to the volume — twenty thousand of them is tens
 * of minutes before the first byte gets written, with nothing printed the whole time.
 * Overlapping the waits is the entire difference.
 */
async function placedSamples(root, concurrency) {
  if (!fs.existsSync(root)) return [];
  const names = fs
    .readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name);

  const progress = track('reading corpus', names.length);
  const broken = [];
  const samples = await mapLimit(names, concurrency, async (name) => {
    const dir = path.join(root, name);
    let raw;
    try {
      raw = await fs.promises.readFile(path.join(dir, meta.FILE), 'utf8');
    } catch (err) {
      // A folder without metadata.json is not a sample, which is what meta.list's own
      // directory filter decided.
      if (err.code !== 'ENOENT') broken.push(`${name}: ${err.message}`);
      progress.tick();
      return null;
    } finally {
      progress.tick();
    }

    try {
      const metadata = JSON.parse(raw);
      return meta.isPlaced(metadata) ? { id: name, dir, metadata } : null;
    } catch (err) {
      // One unreadable sample must not cost the publish of every other one. A container
      // killed mid-write can leave a zero-filled metadata.json behind, and letting that
      // throw took down a run that had 57,590 samples ready to go. It is a sample to fix,
      // named here so it can be, not a reason to stop.
      broken.push(`${name}: ${err.message}`);
      return null;
    }
  });
  progress.done();

  return { samples: samples.filter(Boolean), broken };
}

/** Publishes every placed sample not already up; returns how many went. */
export async function uploadSamples({ force = false } = {}) {
  const { samples: placed, broken } = await placedSamples(GENERATED_DIR, PUBLISH_WIDTH);
  if (broken.length) {
    console.log(`  ${broken.length} sample(s) could not be read and are skipped:`);
    for (const line of broken.slice(0, 10)) console.log(`    ✗ ${line}`);
    if (broken.length > 10) console.log(`    … and ${broken.length - 10} more`);
  }
  if (!placed.length) {
    console.log('  nothing placed to publish');
    return 0;
  }
  console.log(`  ${placed.length} placed sample(s) in the corpus`);

  // Skipping the skip-set is worth it when almost nothing is up yet: computing it walks every
  // published folder, and re-writing two small files over themselves costs less than finding
  // out it was unnecessary.
  const already = force ? new Set() : await timed('walked published folders', () => scene.published());
  const pending = placed.filter((sample) => !already.has(sample.id));
  if (!pending.length) {
    console.log(`  all ${placed.length} placed sample(s) are already up`);
    return 0;
  }

  const batches = [];
  for (let i = 0; i < pending.length; i += PUBLISH_BATCH) batches.push(pending.slice(i, i + PUBLISH_BATCH));

  // A batch fails or lands together, which is the trade for one commit instead of hundreds.
  // Nothing is lost either way: an unpublished sample keeps its metadata and goes next run.
  console.log(`  publishing ${pending.length} sample(s) in ${batches.length} batch(es), ${PUBLISH_WIDTH} wide`);
  const progress = track('publishing', pending.length);
  const results = await mapLimit(batches, PUBLISH_WIDTH, async (batch) => {
    try {
      const { published } = await retry(() => scene.publish(batch.map(payloadOf)));
      progress.tick(batch.length);
      return published.length;
    } catch (err) {
      progress.clear();
      console.error(`    ✗ ${batch.length} sample(s) from ${batch[0].id}: ${err.message}`);
      progress.tick(batch.length);
      return 0;
    }
  });
  progress.done();

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
