// Pulls the small half of published samples off the scene volume into a local corpus, so a
// Modal run can be browsed in the viewer without its meshes ever coming down.
//
//   node scripts/pull-published.mjs                       the 250 most recently published
//   node scripts/pull-published.mjs --limit=0             all of them
//   node scripts/pull-published.mjs --out=generated-modal into a named corpus
//   node scripts/pull-published.mjs --force               re-pull what is already here
//
// Only metadata.json, placement.txt and the two reference images land on disk — a couple of
// hundred KB a sample against the ~18 MB its meshes weigh. The GLBs stay on the volume: the
// server's `/mesh` route falls back to dc-scene-ops for any mesh it cannot find locally, so
// the viewer resolves them one at a time as samples are opened.
//
// Newest first, off the volume's own timestamps, so the default lands on the latest run.
// The corpus name has to start with "generated" or the server's picker will not offer it.

import fs from 'node:fs';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { mapLimit, retry, widthOf } from '../pipeline/limit.mjs';
import * as meta from '../pipeline/metadata.mjs';

const run = promisify(execFile);
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ENV_FILE = path.join(ROOT, '.env');
if (fs.existsSync(ENV_FILE)) process.loadEnvFile(ENV_FILE);

const args = process.argv.slice(2);
const flag = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

const BASE = process.env.SCENE_BASE_URL ?? 'https://starshot-aitools--dc-scene-ops-web.modal.run';
const VOLUME = process.env.SCENE_VOLUME ?? 'trellis-scene-vol-v2';
const PREFIX = process.env.SCENE_PUBLISH_PREFIX ?? 'datasets/raw/stage1';
const MODAL = process.env.TRELLIS_MODAL_BIN ?? 'modal';
const OUT = path.resolve(ROOT, flag('out', 'generated-modal'));
const LIMIT = Number(flag('limit', 250));
const FORCE = args.includes('--force');
// Deliberately narrow. `/file` reloads the volume under a process-wide lock on every request,
// so the endpoint is built for a viewer opening one mesh at a time, not for bulk. Going wide
// makes the service throw 500s and drop connections rather than going faster.
const WIDTH = widthOf('PULL_CONCURRENCY', 4);

const get = (id, name) =>
  retry(async () => {
    const url = `${BASE}/file/${encodeURIComponent(id)}/${encodeURIComponent(name)}`;
    const response = await fetch(url, { signal: AbortSignal.timeout(120_000) });
    if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
    return Buffer.from(await response.arrayBuffer());
  });

/** metadata.json is written last, exactly as publishing writes it: its presence is what marks
 *  the folder complete, so an interrupted pull is retried next run rather than skipped. */
async function pull(id) {
  const dir = path.join(OUT, id);
  fs.mkdirSync(dir, { recursive: true });

  const metadata = await get(id, meta.FILE);
  const parsed = JSON.parse(metadata.toString('utf8'));
  const images = meta.ROLES.map((role) => parsed[role]?.image).filter(Boolean);

  for (const name of ['placement.txt', ...images]) {
    meta.writeAtomic(path.join(dir, name), await get(id, name));
  }
  meta.writeAtomic(path.join(dir, meta.FILE), metadata);
}

/** Published ids newest first. The listing carries the volume's own timestamps, which sort
 *  lexicographically as printed, and it costs one call rather than one per sample. */
async function publishedIds() {
  const { stdout } = await run(MODAL, ['volume', 'ls', VOLUME, PREFIX, '--json'], {
    env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
    maxBuffer: 256 * 1024 * 1024,
  });
  return JSON.parse(stdout)
    .filter((row) => row.Type === 'dir')
    .sort((a, b) => String(b['Created/Modified']).localeCompare(String(a['Created/Modified'])))
    .map((row) => path.posix.basename(row.Filename));
}

const published = await publishedIds();
let ids = FORCE ? published : published.filter((id) => !meta.exists(path.join(OUT, id)));
if (LIMIT > 0) ids = ids.slice(0, LIMIT);

console.log(`\n${published.length} published on the volume, ${ids.length} to pull into ${OUT}`);
if (ids.length) {
  let done = 0;
  const failed = [];
  await mapLimit(ids, WIDTH, async (id) => {
    try {
      await pull(id);
      if (++done % 25 === 0) console.log(`  … ${done}/${ids.length}`);
    } catch (err) {
      failed.push(`${id}: ${err.message}`);
    }
  });
  console.log(`\n${done}/${ids.length} pulled`);
  if (failed.length) {
    console.log(`${failed.length} failed — re-running retries them:`);
    for (const line of failed.slice(0, 10)) console.log(`  ✗ ${line}`);
    if (failed.length > 10) console.log(`  … and ${failed.length - 10} more`);
    process.exitCode = 1;
  }
}

console.log(`\nBrowse them:  npm run dev  →  http://localhost:5173/pipeline.html`);
console.log(`Corpus picker: "${path.basename(OUT)}"  — meshes stream from the volume as you open each sample.`);
