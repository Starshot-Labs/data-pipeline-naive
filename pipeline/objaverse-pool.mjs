// Builds the Objaverse seed pool that scene generation draws its assets from.
//
//   node pipeline/objaverse-pool.mjs                    build the asset list, then tag it
//   node pipeline/objaverse-pool.mjs --limit=800 --tag-limit=240 --no-licenses    smoke test
//
// Two phases, both resumable:
//
//   assets   one row per usable asset out of TRELLIS-500K's curated Sketchfab subset —
//            ~168k objects already filtered to aesthetic score ≥ 5.5, each with captions.
//            The uid inside each row's Sketchfab URL is an Objaverse 1.0 uid, joined here
//            against `object-paths.json.gz` so every row carries the exact GLB path on the
//            Hugging Face repo (which is what the fetch stage will download later), and,
//            unless --no-licenses, against Objaverse's metadata shards for each object's
//            license and artist-given name.
//
//   tagging  an LLM reads each caption and multi-labels the asset with the placement
//            categories it can serve — as the anchor that receives, and as the object that
//            is placed — plus a junk flag for scenes, terrain and fragments. Tagged rows
//            append to pool.jsonl as each batch lands, so a rate limit costs only what was
//            in flight and the next run picks up the untagged remainder.
//
// The category definitions live in scene-spec.mjs, shared with the generator, so what the
// tagger promises is exactly what the generator's prompts assume.

import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chatJSON } from './openrouter.mjs';
import { mapLimit, retry, widthOf } from './limit.mjs';
import { CATEGORIES, CATEGORY_IDS, JUNK } from './scene-spec.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ENV_FILE = path.join(ROOT, '.env');
if (fs.existsSync(ENV_FILE)) process.loadEnvFile(ENV_FILE);

const args = process.argv.slice(2);
const flag = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

const POOL_DIR = path.resolve(ROOT, flag('pool', process.env.POOL_DIR ?? 'pool'));
const CACHE_DIR = path.join(POOL_DIR, 'cache');
const ASSETS_FILE = path.join(POOL_DIR, 'assets.jsonl');
const POOL_FILE = path.join(POOL_DIR, 'pool.jsonl');

const LIMIT = Number(flag('limit', 0));
const TAG_LIMIT = Number(flag('tag-limit', 0));
const WITH_LICENSES = !args.includes('--no-licenses');
const REFRESH = args.includes('--refresh');

const BATCH = Number(process.env.POOL_BATCH ?? 40);
const WIDTH = widthOf('POOL_CONCURRENCY');
// Tagging is plain classification, and the default is the cheapest model that can run with
// thinking fully disabled — the whole Gemini flash family refuses to (OpenRouter answers
// "Reasoning is mandatory"), and mandatory reasoning multiplied the cost of every call
// nearly 20× ($0.07 against luna's measured $0.004).
const MODEL = process.env.POOL_MODEL ?? 'openai/gpt-5.6-luna';
// ...and it must not think: flash models reason by default, which multiplied the cost of
// every call. `default` restores the provider's behaviour, low/medium/high set an effort.
const REASONING = (() => {
  const value = (process.env.POOL_REASONING ?? 'off').trim().toLowerCase();
  if (value === 'default') return undefined;
  return value === 'off' ? { enabled: false } : { effort: value };
})();

const HF = 'https://huggingface.co/datasets';
const TRELLIS_CSV = `${HF}/JeffreyXiang/TRELLIS-500K/resolve/main/ObjaverseXL_sketchfab.csv`;
const OBJECT_PATHS = `${HF}/allenai/objaverse/resolve/main/object-paths.json.gz`;
const METADATA_SHARDS = 160; // metadata/000-000.json.gz … 000-159.json.gz
const shardUrl = (n) => `${HF}/allenai/objaverse/resolve/main/metadata/000-${String(n).padStart(3, '0')}.json.gz`;

// ---------------------------------------------------------------------------- downloads

/** Streams `url` to the cache, once. `maxBytes` asks for just the head of a large file via a
 *  Range request — enough rows for a --limit run without paying for the whole download. */
async function download(url, file, { maxBytes = 0 } = {}) {
  const target = path.join(CACHE_DIR, file);
  if (fs.existsSync(target) && !REFRESH) return target;
  const head = maxBytes ? `${target}.head` : null;
  if (head && fs.existsSync(head) && fs.statSync(head).size >= maxBytes - 1024 && !REFRESH) return head;

  fs.mkdirSync(CACHE_DIR, { recursive: true });
  const controller = new AbortController();
  const response = await fetch(url, {
    signal: controller.signal,
    headers: maxBytes ? { Range: `bytes=0-${maxBytes - 1}` } : {},
  });
  if (!response.ok && response.status !== 206) throw new Error(`${url}: HTTP ${response.status}`);

  const destination = head ?? target;
  const temporary = `${destination}.tmp`;
  const out = fs.createWriteStream(temporary);
  let written = 0;
  try {
    // The server may ignore the Range and answer 200 with the full body; the byte cap below
    // keeps a head request cheap either way.
    for await (const chunk of Readable.fromWeb(response.body)) {
      out.write(chunk);
      written += chunk.length;
      if (maxBytes && written >= maxBytes) {
        controller.abort();
        break;
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') throw err;
  }
  await new Promise((resolve) => out.end(resolve));
  fs.renameSync(temporary, destination);
  console.log(`  ↓ ${file}${head ? ' (head)' : ''}  ${(written / 1e6).toFixed(1)} MB`);
  return destination;
}

const readGz = (file) => zlib.gunzipSync(fs.readFileSync(file)).toString('utf8');

// ------------------------------------------------------------------------------- parsing

/** CSV records off a stream: newlines only split a record outside quotes, so captions that
 *  carry commas (all of them) and stray line breaks survive intact. */
async function* csvRecords(file) {
  let carry = '';
  for await (const chunk of fs.createReadStream(file, 'utf8')) {
    carry += chunk;
    let start = 0;
    let quotes = 0;
    for (let i = 0; i < carry.length; i++) {
      if (carry[i] === '"') quotes++;
      else if (carry[i] === '\n' && quotes % 2 === 0) {
        yield carry.slice(start, i).replace(/\r$/, '');
        start = i + 1;
        quotes = 0;
      }
    }
    carry = carry.slice(start);
  }
  // A head download ends mid-record; whatever is left that parses is a bonus, the rest is
  // dropped by the field-count check in the caller.
  if (carry.trim()) yield carry.replace(/\r$/, '');
}

/** One record into fields, RFC-4180 quoting. */
function csvFields(record) {
  const fields = [];
  let value = '';
  let quoted = false;
  for (let i = 0; i < record.length; i++) {
    const ch = record[i];
    if (quoted) {
      if (ch !== '"') value += ch;
      else if (record[i + 1] === '"') { value += '"'; i++; }
      else quoted = false;
    } else if (ch === '"') quoted = true;
    else if (ch === ',') { fields.push(value); value = ''; }
    else value += ch;
  }
  fields.push(value);
  return fields;
}

const readJsonl = (file) => {
  if (!fs.existsSync(file)) return [];
  const rows = [];
  for (const line of fs.readFileSync(file, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    try { rows.push(JSON.parse(line)); } catch { /* a run cut off mid-append; the row re-tags next run */ }
  }
  return rows;
};

// ------------------------------------------------------------------------------- assets

/** uid → { name, license } for every object, off Objaverse's 160 metadata shards. */
async function loadAnnotations() {
  console.log(`  joining licenses from ${METADATA_SHARDS} metadata shard(s)…`);
  const annotations = new Map();
  await mapLimit(Array.from({ length: METADATA_SHARDS }, (_, n) => n), 8, async (n) => {
    const file = await retry(() => download(shardUrl(n), `metadata-000-${String(n).padStart(3, '0')}.json.gz`));
    for (const [uid, entry] of Object.entries(JSON.parse(readGz(file)))) {
      annotations.set(uid, { name: entry.name ?? null, license: entry.license ?? null });
    }
  });
  return annotations;
}

async function buildAssets() {
  if (fs.existsSync(ASSETS_FILE) && !REFRESH) {
    const rows = readJsonl(ASSETS_FILE);
    console.log(`  ${rows.length} asset(s) already listed — --refresh rebuilds`);
    return rows;
  }

  // ~1.1 KB per CSV row; the slack keeps a --limit run from coming up short.
  const csv = await download(TRELLIS_CSV, 'ObjaverseXL_sketchfab.csv', {
    maxBytes: LIMIT ? Math.ceil(LIMIT * 1500 + 65536) : 0,
  });
  const objectPaths = new Map(Object.entries(JSON.parse(readGz(await download(OBJECT_PATHS, 'object-paths.json.gz')))));
  const annotations = WITH_LICENSES ? await loadAnnotations() : new Map();

  const rows = [];
  let header = true;
  for await (const record of csvRecords(csv)) {
    if (header) { header = false; continue; }
    const fields = csvFields(record);
    if (fields.length !== 4) continue;

    const [, identifier, aesthetic, captionsJson] = fields;
    const uid = identifier.match(/([0-9a-f]{32})\/?$/)?.[1];
    const glb = uid && objectPaths.get(uid);
    if (!glb || !captionsJson) continue;

    let captions;
    try { captions = JSON.parse(captionsJson); } catch { continue; }
    if (!Array.isArray(captions) || !captions.length) continue;

    rows.push({
      uid,
      glb,
      aesthetic: Number(Number(aesthetic).toFixed(3)),
      // Longest caption first in the source; the shortest reads like a name.
      caption: String(captions[0]).trim(),
      short: String(captions.at(-1)).trim(),
      ...(annotations.has(uid) ? annotations.get(uid) : {}),
    });
    if (LIMIT && rows.length >= LIMIT) break;
  }

  fs.mkdirSync(POOL_DIR, { recursive: true });
  const temporary = `${ASSETS_FILE}.tmp`;
  fs.writeFileSync(temporary, rows.map((row) => JSON.stringify(row)).join('\n') + '\n');
  fs.renameSync(temporary, ASSETS_FILE);

  if (WITH_LICENSES) {
    const licenses = {};
    for (const row of rows) licenses[row.license ?? 'unknown'] = (licenses[row.license ?? 'unknown'] ?? 0) + 1;
    console.log(`  licenses: ${Object.entries(licenses).map(([k, v]) => `${k}=${v}`).join('  ')}`);
  }
  console.log(`  ${rows.length} asset(s) listed`);
  return rows;
}

// ------------------------------------------------------------------------------- tagging

const TAG_SYSTEM = `You classify 3D assets for a scene-placement dataset, from their numbered captions. For each numbered asset return its number and:
- junk: true when the caption describes ${JUNK}. Junk assets get empty category lists.
- anchor: every category whose ANCHOR requirement the asset's geometry clearly affords — the anchor is the fixed object that receives a placement.
- placed: every category whose PLACED requirement the asset clearly meets — the placed object is the one being put against the anchor.

Categories:
${CATEGORIES.map((c) => `- ${c.id} (${c.label}): anchor needs ${c.anchor}. placed must be ${c.placed}.`).join('\n')}

Be strict about placed roles — few assets qualify, often none. Tag anchor roles by affordance rather than by typical use: material decides penetrable (wooden furniture qualifies), raised forms decide drapeable-over, and almost every distinct free-standing object can anchor noncontact.`;

// Assets are numbered in the prompt and answer as numbers — echoing 32-hex uids both ways
// cost more tokens than the classification itself.
const TAG_SCHEMA = {
  type: 'object',
  properties: {
    items: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          i: { type: 'integer' },
          junk: { type: 'boolean' },
          anchor: { type: 'array', items: { type: 'string', enum: CATEGORY_IDS } },
          placed: { type: 'array', items: { type: 'string', enum: CATEGORY_IDS } },
        },
        required: ['i', 'junk', 'anchor', 'placed'],
        additionalProperties: false,
      },
    },
  },
  required: ['items'],
  additionalProperties: false,
};

async function tagBatch(batch, spend) {
  const user = batch.map((asset, i) => `${i + 1}. ${asset.caption.slice(0, 240)}`).join('\n');
  const { data, usage } = await chatJSON({
    model: MODEL,
    system: TAG_SYSTEM,
    user,
    name: 'tags',
    schema: TAG_SCHEMA,
    reasoning: REASONING,
  });
  spend.calls++;
  if (typeof usage?.cost === 'number') spend.cost += usage.cost;

  const byIndex = new Map((data.items ?? []).map((item) => [item.i, item]));
  return batch.flatMap((asset, i) => {
    const item = byIndex.get(i + 1);
    if (!item) return []; // dropped by the model; stays pending for the next run
    const valid = (list) => [...new Set(list.filter((id) => CATEGORY_IDS.includes(id)))];
    return [{
      ...asset,
      junk: Boolean(item.junk),
      anchor: item.junk ? [] : valid(item.anchor),
      placed: item.junk ? [] : valid(item.placed),
    }];
  });
}

async function tagAssets(assets) {
  const tagged = new Set(readJsonl(POOL_FILE).map((row) => row.uid));
  let pending = assets.filter((asset) => !tagged.has(asset.uid));
  if (TAG_LIMIT) pending = pending.slice(0, TAG_LIMIT);
  if (!pending.length) {
    console.log(`  all ${assets.length} asset(s) already tagged`);
    return;
  }

  const batches = [];
  for (let i = 0; i < pending.length; i += BATCH) batches.push(pending.slice(i, i + BATCH));
  console.log(`  tagging ${pending.length} asset(s) in ${batches.length} call(s) via ${MODEL}`);

  let done = 0;
  const spend = { calls: 0, cost: 0 };
  await mapLimit(batches, WIDTH, async (batch) => {
    try {
      const rows = await retry(() => tagBatch(batch, spend));
      // One append per batch: a crash costs at most one line, which re-tags next run.
      fs.appendFileSync(POOL_FILE, rows.map((row) => JSON.stringify(row)).join('\n') + '\n');
      done += rows.length;
      if (done % 2000 < BATCH) console.log(`  … ${done}/${pending.length} · $${spend.cost.toFixed(2)}`);
    } catch (err) {
      console.error(`  ✗ batch of ${batch.length}: ${err.message}`);
    }
  });
  console.log(`  tagged ${done}/${pending.length} — ${spend.calls} call(s), $${spend.cost.toFixed(4)}` +
    (spend.calls ? ` ($${(spend.cost / spend.calls).toFixed(5)}/call)` : ''));
}

function summarize() {
  const pool = readJsonl(POOL_FILE);
  const junk = pool.filter((row) => row.junk).length;
  console.log(`\n${pool.length} asset(s) in ${POOL_FILE}, ${junk} junk`);
  for (const category of CATEGORIES) {
    const anchors = pool.filter((row) => row.anchor?.includes(category.id)).length;
    const placed = pool.filter((row) => row.placed?.includes(category.id)).length;
    console.log(`  ${category.id.padEnd(12)} anchors ${String(anchors).padStart(6)}   placed ${String(placed).padStart(6)}`);
  }
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  console.log('\n0 · objaverse seed pool');
  const assets = await buildAssets();
  await tagAssets(assets);
  summarize();
  console.log('\nGenerate scenes with:  node pipeline/generate-scenes.mjs --samples=50000');
}
