// PartField part segmentation, one results folder per mesh.
//
// PartField (github.com/nv-tlabs/PartField) predicts a feature field over a shape and clusters
// the per-triangle features under the mesh's own face adjacency, which gives a part
// decomposition without a prompt, a class list or a second view. It runs on Modal — see
// modal/partfield_app.py — and this drives it over a set of meshes.
//
//   node pipeline/test-segment.mjs                    every anchor in placement-set/
//   node pipeline/test-segment.mjs a.glb dataset/     explicit meshes, or folders of them
//   --parts=8       one cut
//   --parts=8,5,3   the same hierarchy read at three granularities, which is what the right
//                   number of parts being a property of the shape rather than the model makes
//                   worth doing: only the first pays for the clustering
//   --max-clusters=20 --option=0 --with-knn --points-per-face=N --out=DIR --force --concurrency=N
//
// Per mesh, under <out>/<stem>/, per cut:
//   parts_NN.glb   one named, coloured node per part, in the source mesh's own frame
//   labels_NN.bin  little-endian int32 per face, the part face i belongs to
//   parts_NN.png   an isometric look at the cut
// and one segment.json for the job behind them all.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { mapLimit, widthOf } from './limit.mjs';
import { segment, cut, health } from './partfield.mjs';
import { RESULT, writeCut, writeRecord } from './segments.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ENV_FILE = path.join(ROOT, '.env');
if (fs.existsSync(ENV_FILE)) process.loadEnvFile(ENV_FILE);

const args = process.argv.slice(2);
const flag = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const number = (name) => (flag(name) === undefined ? undefined : Number(flag(name)));

const SET_DIR = path.resolve(ROOT, flag('set', 'placement-set'));
const OUT_DIR = path.resolve(ROOT, flag('out', 'segment-results'));
const FORCE = args.includes('--force');
// One in flight is one A10G, and the service holds a job's features and hierarchy on a volume
// until it is discarded, so this is a bill as much as it is a rate.
const WIDTH = Number(flag('concurrency', widthOf('SEGMENT_CONCURRENCY', 2)));

const LEVELS = String(flag('parts', 8)).split(',').map(Number);
const MAX_CLUSTERS = Number(flag('max-clusters', 20));
if (LEVELS.some((level) => !Number.isInteger(level) || level < 1 || level > MAX_CLUSTERS)) {
  throw new Error(`--parts must be whole numbers from 1 to --max-clusters (${MAX_CLUSTERS}), got ${flag('parts', 8)}`);
}

/** Every .glb at or under `target`, so a folder of samples can be named instead of its files. */
function meshesUnder(target) {
  if (fs.statSync(target).isFile()) return [target];
  return fs.readdirSync(target, { withFileTypes: true }).flatMap((entry) => {
    const child = path.join(target, entry.name);
    if (entry.isDirectory()) return meshesUnder(child);
    return entry.name.toLowerCase().endsWith('.glb') ? [child] : [];
  });
}

async function run(file) {
  const id = path.basename(file, path.extname(file));
  const say = (message) => console.log(`  [${id}] ${message}`);

  const out = path.join(OUT_DIR, id);
  if (FORCE) fs.rmSync(out, { recursive: true, force: true });
  if (fs.existsSync(path.join(out, RESULT))) {
    say('already segmented — use --force to redo');
    return;
  }
  fs.mkdirSync(out, { recursive: true });

  const model = fs.readFileSync(file);
  say(`${(model.length / 1e6).toFixed(1)} MB → ${LEVELS.join(', ')} parts`);

  // The first level builds the hierarchy; the rest are read back out of it, so asking for
  // several costs barely more than asking for one.
  const cuts = [];
  let job = null;
  let summary = null;
  for (const parts of LEVELS) {
    const log = (message) => say(`    ${message}`);
    const result = job
      ? await cut({ jobId: job, parts, log })
      : await segment({
          id,
          model,
          filename: path.basename(file),
          parts,
          maxClusters: MAX_CLUSTERS,
          option: number('option'),
          withKnn: args.includes('--with-knn') || undefined,
          pointsPerFace: number('points-per-face'),
          log,
        });
    job = result.jobId;
    summary = result.summary;
    cuts.push(await writeCut(out, result));
  }

  writeRecord(out, { id, source: path.relative(ROOT, file), job, summary, cuts });
  say(`${cuts.map((entry) => entry.parts).join(', ')} parts over ${summary.faces} faces → ${path.relative(ROOT, out)}`);
}

// Read as a folder of meshes rather than through `readSample`, which is the placement
// experiments' reader and insists on a source photo and a phrase that a segmentation has no
// use for — a sample missing either would take the whole run down before it started.
const requested = args.filter((arg) => !arg.startsWith('--'));
const meshes = (requested.length ? requested : [SET_DIR]).flatMap((target) =>
  meshesUnder(path.resolve(ROOT, target)),
);

if (!meshes.length) throw new Error(`no .glb found in ${requested.join(', ') || path.relative(ROOT, SET_DIR)}`);
console.log(`${meshes.length} mesh(es) → ${path.relative(ROOT, OUT_DIR)}`);
console.log(`partfield → ${JSON.stringify(await health())}`);

const results = await mapLimit(meshes, WIDTH, async (file) => {
  try {
    await run(file);
    return true;
  } catch (err) {
    console.error(`  [${path.basename(file)}] ✗ ${err.message}`);
    return false;
  }
});

const done = results.filter(Boolean).length;
console.log(`\n${done}/${meshes.length} segmented`);
if (done < meshes.length) process.exitCode = 1;
