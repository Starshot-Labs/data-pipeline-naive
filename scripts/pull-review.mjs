// Pulls placement verdicts and the frames they were made from down for review.
//
//   node scripts/pull-review.mjs                 everything with a verdict
//   node scripts/pull-review.mjs --limit=50      the first 50
//   node scripts/pull-review.mjs --force         re-pull what is already here
//
// Over HTTP rather than `modal volume get`: the CLI's recursive directory download refuses a
// destination it has just created on Windows, and a per-file `get` is a subprocess each, which
// for fifty samples is four hundred of them. `dc-scene-ops` already serves the review prefix.
//
// The verdict lands last, exactly as publishing writes metadata.json last, so an interrupted
// pull is retried next run rather than read as finished.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { mapLimit, retry, widthOf } from '../pipeline/limit.mjs';
import { track } from '../pipeline/progress.mjs';
import { writeAtomicAsync } from '../pipeline/metadata.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ENV_FILE = path.join(ROOT, '.env');
if (fs.existsSync(ENV_FILE)) process.loadEnvFile(ENV_FILE);

const args = process.argv.slice(2);
const flag = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

const BASE = process.env.SCENE_BASE_URL ?? 'https://starshot-aitools--dc-scene-ops-web.modal.run';
const OUT = path.resolve(ROOT, flag('out', process.env.REVIEW_DIR ?? 'review'));
const LIMIT = Number(flag('limit', 0));
const FORCE = args.includes('--force');
// The route reloads the volume under a process-wide lock per request, so this is a viewer's
// width, not a bulk loader's.
const WIDTH = widthOf('PULL_CONCURRENCY', 6);

const DECISION = 'decision.json';

const get = (id, name) =>
  retry(async () => {
    const url = `${BASE}/review/${encodeURIComponent(id)}/${encodeURIComponent(name)}`;
    const response = await fetch(url, { signal: AbortSignal.timeout(120_000) });
    if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
    return Buffer.from(await response.arrayBuffer());
  });

async function pull(id) {
  const dir = path.join(OUT, id);
  await fs.promises.mkdir(dir, { recursive: true });

  const decision = await get(id, DECISION);
  const views = JSON.parse(decision.toString('utf8')).views ?? [];
  for (const name of views) await writeAtomicAsync(path.join(dir, name), await get(id, name));
  await writeAtomicAsync(path.join(dir, DECISION), decision);
}

const index = await fetch(`${BASE}/review`, { signal: AbortSignal.timeout(300_000) });
if (!index.ok) throw new Error(`review index: HTTP ${index.status}`);
const all = (await index.json()).ids ?? [];

const have = new Set(
  fs.existsSync(OUT)
    ? fs.readdirSync(OUT, { withFileTypes: true }).filter((e) => e.isDirectory() && fs.existsSync(path.join(OUT, e.name, DECISION))).map((e) => e.name)
    : [],
);
let ids = FORCE ? all : all.filter((id) => !have.has(id));
if (LIMIT > 0) ids = ids.slice(0, LIMIT);

console.log(`\n${all.length} verdict(s) on the volume, ${ids.length} to pull into ${OUT}`);
if (ids.length) {
  const progress = track('pulling', ids.length);
  const failed = [];
  await mapLimit(ids, WIDTH, async (id) => {
    try {
      await pull(id);
    } catch (err) {
      failed.push(`${id}: ${err.message}`);
    } finally {
      progress.note(failed.length ? `${failed.length} failed` : '');
      progress.tick();
    }
  });
  progress.done();
  if (failed.length) {
    console.log(`${failed.length} failed — re-running retries them:`);
    for (const line of failed.slice(0, 10)) console.log(`  ✗ ${line}`);
    process.exitCode = 1;
  }
}

console.log(`\nReview them:  npm run dev  →  http://localhost:5173/review.html`);
