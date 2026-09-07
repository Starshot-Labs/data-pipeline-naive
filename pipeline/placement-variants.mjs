// Amendment: each sample's placement phrase in four progressively shortened forms.
//
//   node pipeline/placement-variants.mjs --dry --limit=30   look before committing to 20k
//   node pipeline/placement-variants.mjs                    every published sample
//
// A phrase in this dataset names the placed object first and the anchor second — "cup placed
// on top of the table in the middle". This gives every sample four forms of it: the phrase as
// written, then without the anchor, then without the placed object, then without either, so a
// trained model sees the same pose described at four levels of grounding.
//
// The forms land in the sample itself — `placement` becomes the four, `placement_original`
// keeps the first — and in placement.txt beside it, with a copy in the amendment prefix as the
// standalone artefact that prefix has always been. Both generation paths run this, so it is
// not something anyone should need to remember; it stays runnable on its own for a corpus
// written before that was true, and adopts any amendment already sitting on disk rather than
// paying a model to reword phrases that were already checked.
//
// The variants are deletions, not paraphrases. Every word kept must appear in the original in
// the original order, which is checked rather than trusted — a model that rewrites instead of
// cutting is re-rolled. The one edit allowed beyond deletion is dropping a function word left
// stranded by the removal, and only when it is not part of the spatial relation: "inserted
// into the rim of the wooden bucket" may lose its trailing "of" because that belonged to the
// part reference, but "on top of" and "next to" have to survive intact or the sample means
// nothing.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chatJSON } from './openrouter.mjs';
import { mapLimit, retry, widthOf } from './limit.mjs';
import { track } from './progress.mjs';
import * as meta from './metadata.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ENV_FILE = path.join(ROOT, '.env');
if (fs.existsSync(ENV_FILE)) process.loadEnvFile(ENV_FILE);

const args = process.argv.slice(2);
const flag = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

const SCENE = process.env.SCENE_DIR ?? '/scene';
const STAGING = process.env.SCENE_STAGING_PREFIX ?? 'datasets/raw/staging';
const AMENDMENT = process.env.SCENE_AMENDMENT_PREFIX ?? 'datasets/raw/amendment';
// The staging corpus, not the published one: staging is what every later stage reads and what
// publishing copies from, so variants written anywhere else are variants the dataset will not
// carry — which is exactly how twenty thousand samples ended up amended in a folder nothing
// downstream looked at.
const SOURCE_DIR = path.resolve(ROOT, flag('source', process.env.GENERATED_DIR ?? path.join(SCENE, STAGING)));
const OUT_DIR = path.resolve(ROOT, flag('out', path.join(SCENE, AMENDMENT)));

const FILE = 'placement_variants.txt';
const DRY = args.includes('--dry');
const FORCE = args.includes('--force');
const LIMIT = Number(flag('limit', 0));
const BATCH = Number(process.env.VARIANT_BATCH ?? 25);
const WIDTH = widthOf('VARIANT_CONCURRENCY');
const READ_WIDTH = widthOf('VARIANT_READ_CONCURRENCY', 64);
const MODEL = process.env.VARIANT_MODEL ?? 'openai/gpt-5.6-luna';
const REASONING = (() => {
  const value = (process.env.VARIANT_REASONING ?? 'off').trim().toLowerCase();
  if (value === 'default') return undefined;
  return value === 'off' ? { enabled: false } : { effort: value };
})();

// ------------------------------------------------------------------------------- prompt

const SYSTEM = `You shorten placement phrases by removing object mentions. Each phrase names a placed object first and an anchor object second, and says where the placed object goes relative to the anchor. You get the phrase and both names.

For each numbered phrase return three shortened forms:
- without_anchor: the anchor is gone, the placed object stays.
- without_placed: the placed object is gone, the anchor stays.
- without_both: neither object is named.

Rules:
- Remove the object's WHOLE noun phrase, the article included. Taking "table" out of "on top of the table" leaves "on top" — never "on top of the", never "on top of".
- Every result must read as natural English. No article left with nothing after it, no preposition left hanging.
- Change as little as possible beyond the removal. Keep the original wording and word order, and do not introduce any object, place or detail that was not already there.
- Keep the spatial meaning. The result must still say roughly where the object goes.
- Keep every position, part and manner word: "in the middle", "on the left side", "the fretboard", "folded".

Worked examples — original (placed / anchor):
- "cup placed on top of the table in the middle" (cup / table) → without_anchor "cup placed on top in the middle", without_placed "placed on top of the table in the middle", without_both "placed on top in the middle".
- "bowl leaning against the abacus near the corner" (bowl / abacus) → without_anchor "bowl leaning near the corner". Dropping "against" as well is what keeps it readable.
- "ant floating above the frame of the abacus" (ant / abacus) → without_anchor "ant floating above the frame". The part reference survives; only the anchor and its "of" go.
- "the visor hanging on the 3d object in the middle" (visor / 3d object) → without_placed "hanging on the 3d object in the middle". The placed object's own "the" goes with it.
- "container on top of the air cooler" (container / air cooler) → without_anchor "container on top", without_both "on top".`;

const SCHEMA = {
  type: 'object',
  properties: {
    items: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          i: { type: 'integer' },
          without_anchor: { type: 'string' },
          without_placed: { type: 'string' },
          without_both: { type: 'string' },
        },
        required: ['i', 'without_anchor', 'without_placed', 'without_both'],
        additionalProperties: false,
      },
    },
  },
  required: ['items'],
  additionalProperties: false,
};

// --------------------------------------------------------------------------- validation

const words = (text) => text.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, ' ').split(/\s+/).filter(Boolean);

const ARTICLES = new Set(['the', 'a', 'an']);
// Prepositions and adverbs a removal can strand. An article sitting immediately before one of
// these is the signature of a name lifted out and its article left behind — "leaning against
// the near the corner" — which is the failure this whole check exists to catch.
const FOLLOWERS = new Set([
  'near', 'on', 'in', 'at', 'under', 'over', 'beside', 'between', 'along', 'behind', 'beneath',
  'from', 'to', 'of', 'with', 'by', 'for', 'against', 'into', 'onto', 'above', 'below', 'inside',
]);

const at = (hay, needle, i) => needle.every((word, k) => hay[i + k] === word);

/** Whether `name`'s words appear consecutively in the word list `hay`. */
function mentions(hay, name) {
  const needle = words(name);
  return needle.length ? hay.some((_, i) => at(hay, needle, i)) : false;
}

/**
 * `hay` with every occurrence of `name`'s word sequence taken out.
 *
 * The two object names in a sample overlap more often than you would guess — a sports car
 * leaning against a car, a bed hanging over a canopy bed. Checking that a variant no longer
 * names one of them has to ignore the one it is supposed to keep, or the surviving name's own
 * words read as the removed one still being there.
 */
function without(hay, name) {
  const needle = words(name);
  if (!needle.length) return hay;
  const out = [];
  for (let i = 0; i < hay.length; ) {
    if (at(hay, needle, i)) i += needle.length;
    else out.push(hay[i++]);
  }
  return out;
}

/** A leftover article: trailing, or immediately before a preposition. `named` holds the words
 *  of both object names, because a hyphenated one like "above-ground pool" splits into tokens
 *  that read as prepositions — the article before those belongs to the name, not to a gap. */
function dangling(list, named) {
  if (ARTICLES.has(list.at(-1))) return `ends on "${list.at(-1)}"`;
  for (let i = 0; i < list.length - 1; i++) {
    const next = list[i + 1];
    if (ARTICLES.has(list[i]) && FOLLOWERS.has(next) && !named.has(next)) return `"${list[i]} ${next}"`;
  }
  return null;
}

/**
 * Why a returned set of variants is unusable, or null when it is fine.
 *
 * The variants are allowed to be reworded rather than merely cut — a clean removal often
 * needs a stranded preposition dropped too — so this checks the properties that actually
 * matter instead of demanding the words be a subsequence of the original: the right object is
 * named, the wrong one is not, nothing was invented, and no article was left stranded.
 */
function problemWith(item, { placement, anchor, placed }) {
  const original = words(placement);
  const vocabulary = new Set(original);
  const named = new Set([...words(anchor), ...words(placed)]);
  const checks = [
    ['without_anchor', item.without_anchor, { keep: placed, drop: anchor }],
    ['without_placed', item.without_placed, { keep: anchor, drop: placed }],
    ['without_both', item.without_both, { keep: null, drop: null }],
  ];

  for (const [field, value, { keep, drop }] of checks) {
    if (typeof value !== 'string' || !value.trim()) return `${field} is empty`;
    const list = words(value);
    if (list.length > original.length) return `${field} is longer than the original: "${value}"`;

    // Shortening, not rewriting: a word or two of repair is expected, a new clause is not.
    const novel = list.filter((word) => !vocabulary.has(word));
    if (novel.length > 2) return `${field} invented "${novel.join('", "')}"`;

    const stranded = dangling(list, named);
    if (stranded) return `${field} left a stranded article — ${stranded}: "${value}"`;

    if (keep && !mentions(list, keep)) return `${field} dropped "${keep}"`;
    if (drop && mentions(without(list, keep), drop)) return `${field} still names "${drop}"`;
    if (!keep && (mentions(list, anchor) || mentions(list, placed))) return `${field} still names an object`;
  }
  return null;
}

// ------------------------------------------------------------------------------- corpus

/** Samples with a placement phrase and both object names, read concurrently — a serial walk
 *  over twenty thousand of these is minutes of nothing happening. Samples that already carry
 *  their shortened forms are the finished ones and drop out here. */
async function corpus(root, { concurrency, limit = 0, force = false }) {
  if (!fs.existsSync(root)) throw new Error(`${root} does not exist`);
  const all = fs
    .readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();

  const progress = track('reading corpus', all.length);
  const read = await mapLimit(all, concurrency, async (id) => {
    let raw;
    try {
      raw = await fs.promises.readFile(path.join(root, id, meta.FILE), 'utf8');
    } catch (err) {
      // No metadata.json means the folder is not a sample — not an error, just not ours.
      if (err.code === 'ENOENT') return null;
      throw err;
    } finally {
      progress.tick();
    }
    const metadata = JSON.parse(raw);
    if (!force && meta.hasVariants(metadata)) return { done: true };
    const placement = meta.phraseOf(metadata);
    const anchor = metadata.anchor?.name?.trim();
    const placed = metadata.placed?.name?.trim();
    return placement && anchor && placed
      ? { id, dir: path.join(root, id), metadata, placement, anchor, placed }
      : null;
  });
  progress.done();

  const pending = read.filter((entry) => entry && !entry.done);
  return {
    total: all.length,
    done: read.filter((entry) => entry?.done).length,
    samples: limit ? pending.slice(0, limit) : pending,
  };
}

/**
 * The three shortened forms already sitting in the amendment prefix for `sample`, or null.
 *
 * The first twenty thousand samples were amended before the forms had anywhere to live in
 * the sample itself, so they exist only as loose text files. Re-deriving them from the model
 * would cost real money to produce different wording for phrases that were already checked,
 * so they are adopted verbatim — and still validated, because a file on a volume is not a
 * promise about its contents.
 */
async function adopt(outDir, sample) {
  let text;
  try {
    text = await fs.promises.readFile(path.join(outDir, sample.id, FILE), 'utf8');
  } catch {
    return null;
  }
  const [original, without_anchor, without_placed, without_both] = text.split('\n').map((line) => line.trim());
  if (!without_anchor || !without_placed || !without_both) return null;
  // A phrase that has since been re-written makes its old variants the wrong ones.
  if (original !== sample.placement) return null;
  const item = { without_anchor, without_placed, without_both };
  return problemWith(item, sample) ? null : item;
}

// -------------------------------------------------------------------------------- work

const userFor = (batch) =>
  batch
    .map((s, i) => `${i + 1}. phrase: ${s.placement}\n   placed: ${s.placed}\n   anchor: ${s.anchor}`)
    .join('\n');

async function variantsFor(batch, spend) {
  const { data, usage } = await chatJSON({
    model: MODEL,
    system: SYSTEM,
    user: userFor(batch),
    name: 'placement_variants',
    schema: SCHEMA,
    reasoning: REASONING,
  });
  spend.calls++;
  if (typeof usage?.cost === 'number') spend.cost += usage.cost;

  const byIndex = new Map((data.items ?? []).map((item) => [item.i, item]));
  return batch.map((sample, i) => {
    const item = byIndex.get(i + 1);
    if (!item) return { sample, problem: 'dropped by the model' };
    const problem = problemWith(item, sample);
    return problem ? { sample, problem } : { sample, item };
  });
}

const lines = (sample, item) =>
  [sample.placement, item.without_anchor.trim(), item.without_placed.trim(), item.without_both.trim()].join('\n') + '\n';

const triple = (item) => [item.without_anchor, item.without_placed, item.without_both];

/**
 * Records the forms in both places they belong: the sample itself, which is what every later
 * stage and the published dataset read, and the amendment prefix, which stays as the
 * standalone artefact it has always been.
 *
 * The sample is the one that matters. Writing only beside the dataset was what left twenty
 * thousand samples carrying a phrase in metadata.json and four lines in a folder nothing
 * downstream reads.
 */
async function write(outDir, sample, item, { amendmentExists = false } = {}) {
  meta.setVariants(sample.metadata, triple(item));
  await Promise.all([
    meta.writeAsync(sample.dir, sample.metadata),
    meta.writeAtomicAsync(path.join(sample.dir, 'placement.txt'), meta.placementText(sample.metadata)),
    // An adopted sample's amendment file is already this exact text; rewriting it is a third
    // of the pass's writes spent proving that.
    amendmentExists
      ? Promise.resolve()
      : fs.promises
          .mkdir(path.join(outDir, sample.id), { recursive: true })
          .then(() => meta.writeAtomicAsync(path.join(outDir, sample.id, FILE), lines(sample, item))),
  ]);
}

export async function amend({ sourceDir = SOURCE_DIR, outDir = OUT_DIR, dry = DRY, force = FORCE, limit = LIMIT } = {}) {
  const { total, done: already, samples: pending } = await corpus(sourceDir, {
    concurrency: READ_WIDTH,
    limit,
    force,
  });

  console.log(`  ${total} folder(s), ${already} already carry their variants, ${pending.length} to do`);
  if (!pending.length) return 0;

  // Anything amended before the forms had a home in the sample is adopted rather than
  // re-derived: same wording, no model call, and the twenty thousand that motivated this
  // whole exercise cost nothing.
  let adopted = 0;
  const failures = [];
  const fresh = [];
  // Concurrent: this is one volume read per sample, and twenty thousand of them in series is
  // a quarter of an hour of a stage looking hung before it has done anything.
  const scan = track('adopting existing', pending.length);
  await mapLimit(pending, READ_WIDTH, async (sample) => {
    const item = force ? null : await adopt(outDir, sample);
    if (item) {
      if (dry) console.log(`\n▸ ${sample.id} (adopted)\n${lines(sample, item)}`.trimEnd());
      else await write(outDir, sample, item, { amendmentExists: true });
      adopted++;
    } else {
      fresh.push(sample);
    }
    scan.note(`${adopted} adopted`);
    scan.tick();
  });
  scan.done(`${adopted} adopted, ${fresh.length} need the model`);

  const batches = [];
  for (let i = 0; i < fresh.length; i += BATCH) batches.push(fresh.slice(i, i + BATCH));

  let written = 0;
  const spend = { calls: 0, cost: 0 };
  if (batches.length) {
    console.log(`  writing ${fresh.length} via ${MODEL} in ${batches.length} call(s), ${WIDTH} wide`);
    const progress = track('  variants', fresh.length, { indent: '' });
    await mapLimit(batches, WIDTH, async (batch) => {
      let results;
      try {
        results = await retry(() => variantsFor(batch, spend));
      } catch (err) {
        for (const sample of batch) failures.push(`${sample.id}: ${err.message}`);
        progress.tick(batch.length);
        return;
      }
      for (const { sample, item, problem } of results) {
        if (problem) {
          failures.push(`${sample.id}: ${problem}`);
          continue;
        }
        if (dry) console.log(`\n▸ ${sample.id}\n${lines(sample, item)}`.trimEnd());
        else await write(outDir, sample, item);
        written++;
      }
      progress.note(`$${spend.cost.toFixed(4)}`);
      progress.tick(batch.length);
    });
    progress.done();
  }

  console.log(
    `\n${adopted + written}/${pending.length} ${dry ? 'previewed' : 'written'}` +
      ` — ${adopted} adopted, ${written} generated in ${spend.calls} call(s) to ${MODEL}, $${spend.cost.toFixed(4)}`,
  );
  if (failures.length) {
    console.log(`${failures.length} failed — re-running retries them:`);
    for (const line of failures.slice(0, 15)) console.log(`  ✗ ${line}`);
    if (failures.length > 15) console.log(`  … and ${failures.length - 15} more`);
  }
  return failures.length;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  console.log(`\nplacement variants${DRY ? ' (dry run)' : ''} via ${MODEL}`);
  console.log(`  from ${SOURCE_DIR}`);
  if (!DRY) console.log(`  into ${OUT_DIR}`);
  const failed = await amend();
  if (failed) process.exitCode = 1;
}
