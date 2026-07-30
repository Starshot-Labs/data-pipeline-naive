// Stage 2 of the retrieval pipeline: turn recorded Objaverse uids into meshes and images.
//
//   node pipeline/fetch-assets.mjs                     every sample missing a mesh
//   node pipeline/fetch-assets.mjs <sampleId> ...      specific samples
//   node pipeline/fetch-assets.mjs --source=<dir>      a local corpus — fetches into the
//                                                      sample folders, cache beside the repo
//   node pipeline/fetch-assets.mjs --force             re-fetch even where a mesh is recorded
//
// For each role carrying an `objaverse` block, the GLB is downloaded from Hugging Face and
// a reference image is rendered from it (render.mjs — the software renderer), both through
// a per-uid cache so shared assets cost one download and one render across the whole
// corpus. Volume-resident samples do all of this beside the volume — nothing heavy crosses
// this machine; a `--source` corpus fetches into its own folders for local testing.
//
// Roles without a recorded uid (an invented placed object awaiting retrieval matching) are
// left alone and counted, so the gap is visible rather than silent. `metadata[role].mesh`
// is what marks a role done, exactly as the old mesh stage wrote it, so stages 4-6 run
// unchanged from here.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { mapLimit, widthOf } from './limit.mjs';
import * as scene from './scene.mjs';
import * as meta from './metadata.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ENV_FILE = path.join(ROOT, '.env');
if (fs.existsSync(ENV_FILE)) process.loadEnvFile(ENV_FILE);

const args = process.argv.slice(2);
const flag = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

const LOCAL_SOURCE = flag('source', null);
const SOURCE_DIR = path.resolve(ROOT, LOCAL_SOURCE ?? process.env.GENERATED_DIR ?? 'generated');
const FORCE = args.includes('--force');
// Downloads are ~5-15 MB each; wider mostly queues on the network.
const WIDTH = widthOf('FETCH_CONCURRENCY', 12);

/** The roles of one sample that still need fetching, or null when there is nothing to do. */
function jobsFor(metadata) {
  const stem = meta.stems(metadata);
  const jobs = [];
  let unseeded = 0;
  for (const role of meta.ROLES) {
    const seed = metadata[role].objaverse;
    if (!seed?.uid) unseeded++;
    else if (!metadata[role].mesh || FORCE) jobs.push({ role, stem: stem[role], uid: seed.uid, glb: seed.glb });
  }
  return { jobs, unseeded };
}

export async function fetchSamples({ ids, concurrency = WIDTH } = {}) {
  const samples = meta.list(SOURCE_DIR).filter((sample) => !ids?.length || ids.includes(sample.id));

  let unseededRoles = 0;
  const pending = [];
  for (const sample of samples) {
    const { jobs, unseeded } = jobsFor(sample.metadata);
    unseededRoles += unseeded;
    if (jobs.length) pending.push({ ...sample, jobs });
  }

  console.log(`  ${pending.length} sample(s) to fetch out of ${samples.length}` +
    (unseededRoles ? `  (${unseededRoles} role(s) have no uid and await retrieval matching)` : ''));
  if (!pending.length) return 0;

  let failed = 0;
  await mapLimit(pending, concurrency, async (sample) => {
    try {
      const { fetched, errors } = await scene.fetchAssets(sample.id, sample.jobs, LOCAL_SOURCE ? sample.dir : undefined);

      const metadata = meta.read(sample.dir);
      for (const [role, files] of Object.entries(fetched)) {
        metadata[role].mesh = files.mesh;
        metadata[role].image = files.image;
        metadata[role].textured = true;
      }
      meta.write(sample.dir, metadata);

      for (const { role, uid, error } of errors) {
        failed++;
        console.error(`  ✗ ${sample.id} [${role}] ${uid}: ${error}`);
      }
      const done = Object.keys(fetched);
      if (done.length) console.log(`  + ${sample.id}  ${done.join(' + ')}`);
    } catch (err) {
      failed++;
      console.error(`  ✗ ${sample.id}: ${err.message}`);
    }
  });

  const meshed = meta.list(SOURCE_DIR).filter((sample) => meta.isMeshed(sample.metadata)).length;
  console.log(`\n${meshed} sample(s) now have both meshes.`);
  if (failed) console.log(`${failed} fetch(es) failed — re-running retries them.`);
  return failed;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  console.log(`\n2 · fetching Objaverse assets via ${LOCAL_SOURCE ? `${SOURCE_DIR} (local)` : scene.url()}`);
  const failed = await fetchSamples({ ids: args.filter((arg) => !arg.startsWith('--')) });
  console.log('Place them with:  node pipeline/run.mjs' + (LOCAL_SOURCE ? ` --source=${LOCAL_SOURCE}` : ''));
  if (failed) process.exitCode = 1;
}
