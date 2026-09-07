// Quality filter: does a sample's baked pose actually honour the sentence it was built from?
//
//   node pipeline/judge.mjs --limit=50            judge at most 50 samples; never compile the corpus
//   node pipeline/judge.mjs <id> <id> ...         judge only the named samples
//   node pipeline/judge.mjs --ids-file=<path>     ...or a newline-separated list of them, which
//                                                is how a shard is dealt its slice: eighty
//                                                thousand ids is four megabytes of argv, and
//                                                the exec limit is nearer two
//   node pipeline/judge.mjs --limit=4 --dry       render the views, print the prompt, call nothing
//   node pipeline/judge.mjs --collect             rebuild decisions.json from the verdicts on disk
//   node pipeline/judge.mjs --source=<dir>        judge a local corpus instead of the volume
//
// The placement model reasons over voxel blocks a sixteenth of an object wide, so it is right
// about intent far more often than it is right about geometry — and nothing downstream has
// ever checked. This looks at the result instead of the intent: four cardinal renders from
// 15° above the pair, top and bottom, and an anchor cutaway that exposes containment — all
// untextured with the anchor grey and the placed object red, since which object a surface
// belongs to is the whole question — handed to a vision model with the phrase and names.
//
// Renders come from the posed GLBs themselves, so what is judged is exactly what the dataset
// contains, not a re-derivation of it. Nothing here deletes anything: the verdicts land beside
// the dataset as JSON, and what to do with a `false` stays a separate decision.
//
// Every sample keeps its seven frames next to its verdict, because a filter nobody can audit
// is a filter nobody should trust — `review.html` reads exactly those.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chatJSON } from './openrouter.mjs';
import { mapLimit, retry, widthOf } from './limit.mjs';
import { track } from './progress.mjs';
import { parseGLB, placementOrigin, sceneTriangles } from './glb.mjs';
import { renderViews } from './render.mjs';
import * as meta from './metadata.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ENV_FILE = path.join(ROOT, '.env');
if (fs.existsSync(ENV_FILE)) process.loadEnvFile(ENV_FILE);

const args = process.argv.slice(2);
const flag = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

const SCENE = process.env.SCENE_DIR ?? '/scene';
const PUBLISH = process.env.SCENE_PUBLISH_PREFIX ?? 'datasets/raw/stage1';
const REVIEW = process.env.SCENE_REVIEW_PREFIX ?? 'datasets/raw/review';
// The published corpus, because that is where the posed meshes are.
const SOURCE_DIR = path.resolve(ROOT, flag('source', path.join(SCENE, PUBLISH)));
const OUT_DIR = path.resolve(ROOT, flag('out', path.join(SCENE, REVIEW)));

const DRY = args.includes('--dry');
const FORCE = args.includes('--force');
const LIMIT = Number(flag('limit', 0));

const MODEL = process.env.JUDGE_MODEL ?? 'openai/gpt-5.6-luna';
// Minimal by default: the question is a visual one, and paying for a long chain of thought
// per sample across eighty thousand of them is the difference between tens and hundreds of
// dollars. `off`, `low`, `medium`, `high` and `default` all work if that proves too little.
const REASONING = (() => {
  const value = (process.env.JUDGE_REASONING ?? 'minimal').trim().toLowerCase();
  if (value === 'default') return undefined;
  return value === 'off' ? { enabled: false } : { effort: value };
})();
// Narrower than the text stages: seven images a request is a far larger upload, and the
// per-sample render is CPU work this process does itself.
const WIDTH = widthOf('JUDGE_CONCURRENCY', 16);
// Smaller than a reference image on purpose. The question is whether a mug is inside a bowl
// or floating above it, which survives downsampling; seven 512² frames a call would triple
// the token bill to answer it no better.
const VIEW_SIZE = Number(process.env.JUDGE_VIEW_SIZE ?? 384);

const FILE = 'decision.json';

// ------------------------------------------------------------------------------- prompt

const SYSTEM =
  'You check 3D object placements for a dataset. You are shown seven untextured views and ' +
  'world-space geometry for two posed meshes: the anchor A is grey and the placed object B ' +
  'is red. Judge whether the requested arrangement is clearly correct.';

/** The visual question and the exact world-space geometry of the posed pair. */
export const promptFor = ({ placed, anchor, placement, bounding_box_a, origin_a, bounding_box_b, origin_b }) =>
  [
    'You are given 7 images of the pair. Images 1-4 are from each cardinal side, image 5 is from the top, and image 6 is from the bottom.',
    'Image 7 repeats the front view with the camera-facing half of A clipped away at its center; B remains whole. Treat it as a cross-sectional view, not missing or broken geometry, and use it to see whether B is inside A.',
    `A is the anchor (${anchor}); B is the placed object (${placed}).`,
    'A is grey and B is red.',
    'Here are the world-space bounding boxes and origins of both A and B:',
    JSON.stringify({ bounding_box_a, origin_a, bounding_box_b, origin_b }, null, 2),
    '',
    `B was placed relative to A by an LLM. From the images and geometry, decide whether the objects are UNMISTAKEABLY: a(n) "${placement}".`,
    'If the placement description is PERFECTLY honored and coherent in the 3D models, output true; otherwise output false and this pair will be deleted from the dataset.',
    'Assume the data is bad unless the evidence shows it is good, we only want perfect data.',
  ].join('\n');

// `reason` is not asked for by the filter itself — it exists so a human reviewing the
// verdicts can see what the model thought it was looking at, which is the only way to tell a
// good filter from a confident one.
const SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'boolean' },
    reason: { type: 'string' },
  },
  required: ['verdict', 'reason'],
  additionalProperties: false,
};

// ------------------------------------------------------------------------------- corpus

/**
 * Published samples with both posed meshes on disk and a phrase to judge them against.
 *
 * `only` names them explicitly, which is what a shard passes: several containers all picking
 * "whatever has no verdict yet" off one volume would race for the same samples and render
 * them twice.
 */
async function corpus(root, { force, outDir, only = null }) {
  if (!fs.existsSync(root)) throw new Error(`${root} does not exist`);
  const ids = only?.length
    ? [...only]
    : fs
        .readdirSync(root, { withFileTypes: true })
        .filter((entry) => entry.isDirectory())
        .map((entry) => entry.name)
        .sort();

  const progress = track('reading corpus', ids.length);
  const read = await mapLimit(ids, widthOf('CORPUS_CONCURRENCY', 64), async (id) => {
    try {
      if (!force && (await fs.promises.access(path.join(outDir, id, FILE)).then(() => true, () => false))) {
        return { done: true };
      }
      const raw = await fs.promises.readFile(path.join(root, id, meta.FILE), 'utf8');
      const metadata = JSON.parse(raw);
      const dir = path.join(root, id);
      const meshes = meta.ROLES.map((role) => metadata[role]?.mesh);
      const sample = {
        id,
        dir,
        meshes,
        placed: metadata.placed?.name ?? '',
        anchor: metadata.anchor?.name ?? '',
        placement: meta.phraseOf(metadata),
      };
      if (!meshes.every(Boolean) || !sample.placed || !sample.anchor || !sample.placement) {
        return { ...sample, incomplete: 'metadata names no mesh for both roles' };
      }
      // A published folder is supposed to hold its posed meshes; some hold only the two text
      // files. There is nothing to render and nothing a phrase could be honoured by, so it is
      // settled here rather than sent to the model to fail three times over a missing file.
      const present = await Promise.all(
        meshes.map((mesh) => fs.promises.access(path.join(dir, mesh)).then(() => true, () => false)),
      );
      if (!present.every(Boolean)) {
        return { ...sample, incomplete: 'posed meshes missing from the published folder' };
      }
      return sample;
    } catch {
      // A folder that is not a readable sample is not this stage's problem to report.
      return null;
    } finally {
      progress.tick();
    }
  });
  progress.done();

  const pending = read.filter((entry) => entry && !entry.done);
  return { total: ids.length, done: read.filter((e) => e?.done).length, samples: pending };
}

// --------------------------------------------------------------------------------- work

const rounded = (values) => values.map((value) => Number(value.toFixed(6)));

/** World-space size and placement origin of a posed mesh. */
function geometryOf(glb) {
  const triangles = sceneTriangles(glb);
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < triangles.length; i += 3) {
    for (let axis = 0; axis < 3; axis++) {
      min[axis] = Math.min(min[axis], triangles[i + axis]);
      max[axis] = Math.max(max[axis], triangles[i + axis]);
    }
  }
  const origin = placementOrigin(glb);
  if (!origin) throw new Error('posed mesh carries no placement origin');
  return {
    boundingBox: rounded(max.map((value, axis) => value - min[axis])),
    origin: rounded(origin),
  };
}

/** The seven frames and geometry of a sample's posed pair, in anchor/A then placed/B order. */
async function evidenceOf(sample) {
  const glbs = sample.meshes.map((mesh) => parseGLB(fs.readFileSync(path.join(sample.dir, mesh))));
  const [a, b] = glbs.map(geometryOf);
  return {
    views: await renderViews(glbs, { size: VIEW_SIZE }),
    geometry: {
      bounding_box_a: a.boundingBox,
      origin_a: a.origin,
      bounding_box_b: b.boundingBox,
      origin_b: b.origin,
    },
  };
}

async function judgeOne(sample, spend) {
  const { views, geometry } = await evidenceOf(sample);
  const prompt = promptFor({ ...sample, ...geometry });

  const { data, usage } = await chatJSON({
    model: MODEL,
    system: SYSTEM,
    user: prompt,
    images: views.map((view) => view.png),
    name: 'placement_verdict',
    schema: SCHEMA,
    reasoning: REASONING,
  });
  spend.calls++;
  if (typeof usage?.cost === 'number') spend.cost += usage.cost;

  if (typeof data.verdict !== 'boolean') throw new Error(`no verdict in ${JSON.stringify(data).slice(0, 200)}`);
  return { views, prompt, geometry, verdict: data.verdict, reason: (data.reason ?? '').trim() };
}

/** The verdict and the frames it was made from, so it can be audited rather than believed. */
async function write(outDir, sample, { views, geometry = {}, verdict, reason }) {
  const dir = path.join(outDir, sample.id);
  await fs.promises.mkdir(dir, { recursive: true });
  await Promise.all([
    ...views.map((view, i) => meta.writeAtomicAsync(path.join(dir, `${i + 1}-${view.name}.png`), view.png)),
    meta.writeAtomicAsync(
      path.join(dir, FILE),
      `${JSON.stringify(
        {
          id: sample.id,
          verdict,
          reason,
          placement: sample.placement,
          placed: sample.placed,
          anchor: sample.anchor,
          ...geometry,
          model: MODEL,
          at: new Date().toISOString(),
          views: views.map((view, i) => `${i + 1}-${view.name}.png`),
        },
        null,
        2,
      )}\n`,
    ),
  ]);
}

/** Every verdict on disk, collected into one file so a consumer needs one read, not 80,000. */
export async function compile(outDir) {
  if (!fs.existsSync(outDir)) return { decisions: {}, kept: 0, dropped: 0 };
  const ids = fs
    .readdirSync(outDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();

  const progress = track('collecting verdicts', ids.length);
  const rows = await mapLimit(ids, widthOf('CORPUS_CONCURRENCY', 64), async (id) => {
    try {
      return JSON.parse(await fs.promises.readFile(path.join(outDir, id, FILE), 'utf8'));
    } catch {
      return null;
    } finally {
      progress.tick();
    }
  });
  progress.done();

  const decisions = {};
  let kept = 0;
  for (const row of rows) {
    if (!row) continue;
    decisions[row.id] = {
      verdict: row.verdict,
      reason: row.reason,
      placement: row.placement,
      ...(row.views?.length ? {} : { unseen: true }),
    };
    if (row.verdict) kept++;
  }
  const total = Object.keys(decisions).length;
  const summary = { model: MODEL, at: new Date().toISOString(), total, kept, dropped: total - kept, decisions };
  await meta.writeAtomicAsync(path.join(outDir, 'decisions.json'), `${JSON.stringify(summary, null, 2)}\n`);
  return summary;
}

export async function judgeSamples({ sourceDir = SOURCE_DIR, outDir = OUT_DIR, dry = DRY, force = FORCE, limit = LIMIT, ids = null } = {}) {
  const { total, done, samples: pending } = await corpus(sourceDir, { force, outDir, only: ids });
  const samples = ids?.length || !limit ? pending : pending.slice(0, limit);
  console.log(`  ${total} published sample(s), ${done} already judged, ${samples.length} to do`);
  if (!samples.length) return { failures: 0 };

  if (dry) {
    for (const sample of samples) {
      const { views, geometry } = await evidenceOf(sample);
      const dir = path.join(outDir, sample.id);
      fs.mkdirSync(dir, { recursive: true });
      for (const [i, view] of views.entries()) {
        fs.writeFileSync(path.join(dir, `${i + 1}-${view.name}.png`), view.png);
      }
      console.log(`\n▸ ${sample.id}  (${views.length} views → ${dir})\n${promptFor({ ...sample, ...geometry })}`);
    }
    return { failures: 0 };
  }

  console.log(`  judging ${samples.length} via ${MODEL} (${JSON.stringify(REASONING)}), ${WIDTH} wide`);
  const progress = track('judged', samples.length);
  const spend = { calls: 0, cost: 0 };
  const failures = [];
  let kept = 0;
  let dropped = 0;

  let incomplete = 0;
  await mapLimit(samples, WIDTH, async (sample) => {
    try {
      if (sample.incomplete) {
        // Recorded as a verdict, not skipped: "this sample cannot be used" is exactly the
        // answer the filter exists to produce, and it should not be re-asked every run.
        await write(outDir, sample, { views: [], verdict: false, reason: sample.incomplete });
        incomplete++;
        dropped++;
        return;
      }
      const result = await retry(() => judgeOne(sample, spend));
      await write(outDir, sample, result);
      if (result.verdict) kept++;
      else dropped++;
    } catch (err) {
      failures.push(`${sample.id}: ${err.message}`);
    } finally {
      progress.note(`${kept} keep / ${dropped} drop · $${spend.cost.toFixed(4)}`);
      progress.tick();
    }
  });
  progress.done();
  if (incomplete) console.log(`  ${incomplete} sample(s) dropped unseen — no posed meshes published`);

  console.log(
    `\n  ${kept + dropped}/${samples.length} judged — ${kept} keep, ${dropped} drop` +
      ` — ${spend.calls} call(s) to ${MODEL}, $${spend.cost.toFixed(4)}`,
  );
  if (failures.length) {
    console.log(`  ${failures.length} failed — re-running retries them:`);
    for (const line of failures.slice(0, 10)) console.log(`    ✗ ${line}`);
    if (failures.length > 10) console.log(`    … and ${failures.length - 10} more`);
  }
  return { failures: failures.length };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (args.includes('--collect')) {
    const summary = await compile(OUT_DIR);
    console.log(`\n  ${summary.total} verdict(s): ${summary.kept} keep, ${summary.dropped} drop → decisions.json`);
  } else {
    const idsFile = flag('ids-file', null);
    const ids = idsFile
      ? fs.readFileSync(idsFile, 'utf8').split('\n').map((line) => line.trim()).filter(Boolean)
      : args.filter((arg) => !arg.startsWith('--'));
    console.log(`\nplacement review${DRY ? ' (dry run)' : ''} via ${MODEL}`);
    console.log(`  from ${SOURCE_DIR}`);
    console.log(`  into ${OUT_DIR}`);
    const bounded = ids.length > 0 || LIMIT > 0;
    const { failures } = await judgeSamples({ ids: ids.length ? ids : null });
    if (!DRY && !bounded) {
      const summary = await compile(OUT_DIR);
      console.log(`\n  ${summary.total} verdict(s): ${summary.kept} keep, ${summary.dropped} drop → decisions.json`);
    }
    if (failures) process.exitCode = 1;
  }
}
