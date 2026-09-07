// Backfill: new placement relationships dealt across object pairs the corpus already has.
//
//   node pipeline/backfill-relations.mjs --add=60000        grow the corpus by 60k samples
//   node pipeline/backfill-relations.mjs --samples=80000    ...or toward a total
//   node pipeline/backfill-relations.mjs --add=120 --dry    print the plan and one prompt
//                                                           per category, call nothing
//
// A corpus where every pair appears once teaches "these two objects go like this" and lets
// the phrase go unread. This stage re-reads the pairs the corpus already has — both uids,
// the validated names and descriptions, every phrase ever written for them — and deals each
// pair new (category, relation, detail) assignments under the same weighted quotas
// generation used, so the apple that only ever sat on the right of the desk shows up on its
// left, under it, floating above it. Each new sample is a fresh folder carrying the same
// seeds: the fetch stage hits the per-uid cache and stages 2-6 run unchanged.
//
// Eligibility is tiered rather than binary. `natural` is the strict test generation applied
// when it invented pairs: the anchor's pool tags afford the category and the placed
// object's tags qualify for it. `stretch` loosens each category the one way that stays
// defensible — anything may rest against a rigid anchor (a soft object dealt "on top of"
// still poses right, because the placement model classifies the contact itself and calls it
// a drape), anything may hover, containment and bonded fall back to a size test against the
// pair's baked sizes, and penetrative admits blunt objects but only as "pushed into". Soft
// is never loosened on the placed side: the cloth solver would happily melt a mug over a
// chair, and that is data poison, not data. The dealer spends natural capacity first and
// reaches for stretch only where a quota cannot fill without it; the writer model then
// judges every assignment itself (`fit`), and a verdict of nonsense retires that
// pair-category for the rest of the run.
//
// Re-running is safe: backfill samples carry a `pair` key, count toward the same quotas,
// and their phrases join the very lists new phrases are checked against, so a second pass
// fills gaps rather than doubling anything.

import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chatJSON } from './openrouter.mjs';
import { mapLimit, retry, widthOf } from './limit.mjs';
import { CATEGORIES, DETAIL_LINES } from './scene-spec.mjs';
import { track, duration } from './progress.mjs';
import * as meta from './metadata.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ENV_FILE = path.join(ROOT, '.env');
if (fs.existsSync(ENV_FILE)) process.loadEnvFile(ENV_FILE);

const args = process.argv.slice(2);
const flag = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

const GENERATED_DIR = path.resolve(ROOT, flag('out', process.env.GENERATED_DIR ?? 'generated'));
const POOL_FILE = path.join(path.resolve(ROOT, flag('pool', process.env.POOL_DIR ?? 'pool')), 'pool.jsonl');
const DRY = args.includes('--dry');
// `--pairs=N` works on the first N pairs by sorted key rather than a sample. Deterministic
// on purpose: a smoke test that picked randomly would choose a different N every time it
// resumed, so a crash would scatter half-finished work across the corpus instead of
// finishing the pairs it started. Sorted uid order is arbitrary with respect to category,
// so the slice is representative without being random.
const PAIR_LIMIT = Number(flag('pairs', 0));

// Writing a phrase for a fixed pair is the same job as generation minus the choosing, so
// the writer defaults to whatever generation uses, knobs included.
const MODEL = process.env.BACKFILL_MODEL ?? process.env.SPEC_MODEL ?? 'google/gemini-3.6-flash';
const REASONING = (() => {
  const value = (process.env.BACKFILL_REASONING ?? process.env.SPEC_REASONING ?? 'low').trim().toLowerCase();
  return value === 'off' ? { enabled: false } : { effort: value };
})();
const BATCH = Number(process.env.BACKFILL_BATCH ?? 16);
const WIDTH = widthOf('BACKFILL_CONCURRENCY');

// How many backfills one pair may carry, in total and per category. The total keeps quota
// pressure from saturating a pair — five towels on one couch is one pair memorised five
// ways. Soft and bonded run one higher because their capacity is structurally thin (soft
// placed objects are ~3% of the pool, bonded ~0.2%) and their intra-category variety is
// real: soft has three relations that read differently, bonded five.
const MAX_PER_PAIR = Number(process.env.BACKFILL_MAX_PER_PAIR ?? 5);
const CATEGORY_CAP = { rigid: 2, soft: 3, penetrative: 2, containment: 2, bonded: 3, noncontact: 2 };

const PHRASE_MAX_WORDS = 15;
// Mirrors generate-scenes.mjs — checked with the object names stripped out first, so a
// "tent stake" never trips over its own name.
const BANNED_VERBS = /\b(stab|stuck|stick|jam|shov|wedg|cram|smash|slam|stak|pierc|embed)\w*/i;

const words = (text) => text.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, ' ').split(/\s+/).filter(Boolean);
const at = (hay, needle, i) => needle.every((word, k) => hay[i + k] === word);

/** `hay` with every run of `phrase`'s words removed — placement-variants' trick, for the
 *  same reason: comparing two phrases has to ignore the words that are only there because
 *  the names and the relation are. */
function without(hay, phrase) {
  const needle = words(phrase);
  if (!needle.length) return hay;
  const out = [];
  for (let i = 0; i < hay.length; ) {
    if (at(hay, needle, i)) i += needle.length;
    else out.push(hay[i++]);
  }
  return out;
}

function shuffled(items) {
  const deck = [...items];
  for (let i = deck.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [deck[i], deck[j]] = [deck[j], deck[i]];
  }
  return deck;
}

// ---------------------------------------------------------------------------------- quotas

const COMPLEXITIES = ['simple', 'complex'];
const keyOf = (categoryId, complexity) => `${categoryId}|${complexity}`;

/**
 * `total` split across `categories` in proportion to their weights, the fractional
 * remainder going to the categories closest to their next whole sample, each category then
 * halved between the phrase styles with simple taking the odd one. Over all of CATEGORIES
 * this is exactly generate-scenes' quota split, so a backfilled corpus lands on the same
 * mix a generated one would; over a subset it is how a shortfall no category can serve gets
 * spread across the ones that still have room.
 */
function spread(total, categories) {
  const totalWeight = categories.reduce((sum, c) => sum + c.weight, 0);
  const shares = categories.map((c) => ({ id: c.id, exact: (total * c.weight) / totalWeight }));
  const perCategory = Object.fromEntries(shares.map((s) => [s.id, Math.floor(s.exact)]));
  let left = total - Object.values(perCategory).reduce((sum, q) => sum + q, 0);
  for (const { id } of [...shares].sort((a, b) => (b.exact % 1) - (a.exact % 1))) {
    if (left-- <= 0) break;
    perCategory[id]++;
  }

  const quotas = {};
  for (const category of categories) {
    const quota = perCategory[category.id];
    quotas[keyOf(category.id, 'simple')] = Math.ceil(quota / 2);
    quotas[keyOf(category.id, 'complex')] = Math.floor(quota / 2);
  }
  return quotas;
}

// ----------------------------------------------------------------------------------- tiers

const byLength = (size) => [...size].sort((a, b) => a - b);
const maxDim = (size) => Math.max(...size);

/** Whether the placed mesh, at the relative scale the pair's original bake settled on,
 *  fits within `factor` of the anchor per sorted axis. The placement model re-chooses B's
 *  scale anyway, so this is a semantic guard — no wardrobes inside mugs — not a geometric
 *  promise. Pairs never placed have no sizes and fail it. */
function fitsWithin(sizes, factor) {
  if (!sizes) return false;
  const anchor = byLength(sizes.anchor);
  const placed = byLength(sizes.placed);
  return placed.every((v, i) => v <= anchor[i] * factor);
}

/**
 * 'natural' | 'stretch' | null for one pair and category.
 *
 * natural is exactly the test generation applied when it invented pairs: the anchor's tags
 * afford the category and the placed object's tags qualify for it, with noncontact anchors
 * needing no affordance at all (matching objaverse-pool's loadPool). stretch is the
 * documented loosening, one rule per category, and null is never dealt.
 */
function tierFor(category, pair) {
  const anchorTags = pair.tags.anchor?.anchor ?? [];
  const placedTags = pair.tags.placed?.placed ?? [];
  const anchorOk = category.id === 'noncontact' || anchorTags.includes(category.id);
  if (anchorOk && placedTags.includes(category.id)) return 'natural';

  switch (category.id) {
    case 'noncontact':
      // Hovering asks nothing of either object.
      return 'stretch';
    case 'rigid':
      // Anything may rest against a rigid-affording anchor. A soft placed object still
      // poses right: the placement model classifies the contact itself, so a shirt dealt
      // "on top of" comes back a drape.
      return anchorOk ? 'stretch' : null;
    case 'soft':
      // The placed side is never loosened — rigid bodies do not drape. The anchor side is:
      // fabric lies over almost anything that anchors something.
      return placedTags.includes('soft') && anchorTags.length ? 'stretch' : null;
    case 'containment':
      return anchorOk && fitsWithin(pair.sizes, 0.9) ? 'stretch' : null;
    case 'penetrative':
      // Blunt objects go in too — soil, foam and straw take a push — but only as "pushed
      // into"; "inserted into" stays for the pointed. The dealer forces the relation.
      return anchorOk && pair.sizes && maxDim(pair.sizes.placed) <= maxDim(pair.sizes.anchor) * 0.7
        ? 'stretch'
        : null;
    case 'bonded':
      // Small and light enough to be pinned or clipped to a face without gravity's help.
      return anchorOk && pair.sizes && maxDim(pair.sizes.placed) <= maxDim(pair.sizes.anchor) * 0.4
        ? 'stretch'
        : null;
    default:
      return null;
  }
}

// ----------------------------------------------------------------------------------- pairs

function loadTags() {
  if (!fs.existsSync(POOL_FILE)) {
    throw new Error(`${POOL_FILE} not found — the pool the corpus was seeded from`);
  }
  const started = Date.now();
  const tags = new Map();
  // One large sequential read rather than many small ones, so this stays fast on a volume.
  for (const line of fs.readFileSync(POOL_FILE, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    try {
      const row = JSON.parse(line);
      tags.set(row.uid, row);
    } catch { /* a line cut off mid-append */ }
  }
  console.log(`  ✓ pool tags ${tags.size.toLocaleString('en-US')} asset(s) · ${duration(Date.now() - started)}`);
  return tags;
}

const roleOf = ({ name, description, objaverse }) => ({
  name: name.trim(),
  description: (description ?? '').trim(),
  objaverse,
});

/**
 * One entry per (anchor uid, placed uid) couple, aggregated over every sample that pairs
 * them: names and descriptions from the first-seen sample (already validated at its birth),
 * every phrase ever written for the pair — originals and previous backfills alike, since
 * the novelty checks run against all of them — how many backfills it already carries per
 * category, and the baked sizes for the tier tests where a placed sample provides them.
 */
function harvest(samples, tags) {
  const pairs = new Map();
  let skipped = 0;
  for (const metadata of samples) {
    const anchorUid = metadata.anchor?.objaverse?.uid;
    const placedUid = metadata.placed?.objaverse?.uid;
    if (!anchorUid || !placedUid || !metadata.category) {
      skipped++;
      continue;
    }

    const key = `${anchorUid}|${placedUid}`;
    let pair = pairs.get(key);
    if (!pair) {
      pair = {
        key,
        context: metadata.context,
        anchor: roleOf(metadata.anchor),
        placed: roleOf(metadata.placed),
        tags: { anchor: tags.get(anchorUid), placed: tags.get(placedUid) },
        sizes: null,
        existing: [], // every sample of the pair: { category, relation, detail, phrase }
        backfills: 0, // samples carrying a `pair` key
        perCategory: Object.fromEntries(CATEGORIES.map((c) => [c.id, 0])),
        tier: null,
      };
      pairs.set(key, pair);
    }

    pair.existing.push({
      category: metadata.category,
      relation: metadata.relation ?? '',
      detail: metadata.detail ?? '',
      phrase: meta.phraseOf(metadata),
    });
    if (metadata.pair) {
      pair.backfills++;
      if (metadata.category in pair.perCategory) pair.perCategory[metadata.category]++;
    }
    if (!pair.sizes && metadata.anchor.size && metadata.placed.size) {
      pair.sizes = { anchor: metadata.anchor.size, placed: metadata.placed.size };
    }
  }

  for (const pair of pairs.values()) {
    pair.tier = Object.fromEntries(CATEGORIES.map((c) => [c.id, tierFor(c, pair)]));
  }
  return { pairs: [...pairs.values()], skipped };
}

// ------------------------------------------------------------------------------------ plan

/**
 * Deals the shortfall across the pairs: repeated passes, each giving a pair at most one new
 * assignment, so counts stay even instead of the first pairs in the walk soaking up
 * everything. Within a pair the scarcest eligible category wins — bonded and soft fill from
 * the few pairs that can serve them before rigid takes those pairs' remaining room — with
 * natural capacity spent before stretch, and a pair's first backfill preferring a category
 * its existing samples do not cover, which is what buys cross-category variety per pair.
 */
function deal(shortfalls, pairs) {
  const remaining = { ...shortfalls };
  const left = () => Object.values(remaining).reduce((n, v) => n + v, 0);
  const demand = (id) => (remaining[keyOf(id, 'simple')] ?? 0) + (remaining[keyOf(id, 'complex')] ?? 0);

  // Static scarcity: how much of the run's demand each category can actually serve.
  const capacity = {};
  for (const category of CATEGORIES) {
    capacity[category.id] = 0;
    for (const pair of pairs) {
      if (pair.tier[category.id]) {
        capacity[category.id] += Math.max(0, CATEGORY_CAP[category.id] - pair.perCategory[category.id]);
      }
    }
  }
  const scarcity = (id) => demand(id) / Math.max(1, capacity[id]);

  // What this pass has already reserved, on top of the live counts writes maintain.
  const planned = new Map();
  const stateOf = (pair) => {
    let state = planned.get(pair.key);
    if (!state) {
      state = { total: 0, perCategory: {}, bare: new Set(), dealt: new Set() };
      planned.set(pair.key, state);
    }
    return state;
  };

  // Relations and detail levels rotate per (category, style) with a random start, exactly
  // as generation deals them.
  const cursors = {};
  const dealKnobs = (category, complexity) => {
    const kind = keyOf(category.id, complexity);
    let cursor = cursors[kind];
    if (!cursor) {
      const relations = category[complexity].relations;
      const levels = complexity === 'simple' ? ['bare', 'position'] : ['position', 'part'];
      cursor = cursors[kind] = {
        relations,
        levels,
        rel: Math.floor(Math.random() * relations.length),
        lvl: Math.floor(Math.random() * levels.length),
      };
    }
    return {
      relation: cursor.relations[cursor.rel++ % cursor.relations.length],
      level: cursor.levels[cursor.lvl++ % cursor.levels.length],
    };
  };

  const assignments = [];
  let progress = true;
  while (progress && left() > 0) {
    progress = false;
    for (const pair of shuffled(pairs)) {
      if (left() <= 0) break;
      const state = stateOf(pair);
      if (pair.backfills + state.total >= MAX_PER_PAIR) continue;

      let candidates = CATEGORIES.filter((category) => {
        const used = pair.perCategory[category.id] + (state.perCategory[category.id] ?? 0);
        return pair.tier[category.id] && used < CATEGORY_CAP[category.id] && demand(category.id) > 0;
      });
      if (!candidates.length) continue;

      // A pair's first backfill goes to a category it has not seen, when one is eligible —
      // one genuinely different pose mode per pair before anything else.
      if (pair.backfills + state.total === 0) {
        const seen = new Set(pair.existing.map((entry) => entry.category));
        const fresh = candidates.filter((category) => !seen.has(category.id));
        if (fresh.length) candidates = fresh;
      }

      const pick = (tier) =>
        candidates
          .filter((category) => pair.tier[category.id] === tier)
          .sort((a, b) => scarcity(b.id) - scarcity(a.id))[0];
      const category = pick('natural') ?? pick('stretch');
      if (!category) continue;
      const tier = pair.tier[category.id];

      const complexity = COMPLEXITIES.filter((cx) => (remaining[keyOf(category.id, cx)] ?? 0) > 0)
        .sort((a, b) => remaining[keyOf(category.id, b)] - remaining[keyOf(category.id, a)])[0];

      // A bare phrase is the names and the relation and nothing else, so a second bare of
      // the same relation would be the same sentence. It upgrades to a position.
      const effectiveLevel = (relation, level) => {
        const bareTaken =
          state.bare.has(relation) ||
          pair.existing.some((entry) => entry.relation === relation && entry.detail === 'bare');
        return level === 'bare' && bareTaken ? 'position' : level;
      };

      let { relation, level: rawLevel } = dealKnobs(category, complexity);
      let level = effectiveLevel(relation, rawLevel);
      if (category.id === 'penetrative' && tier === 'stretch') {
        // Stretch penetrative is blunt-into-soft-material, which only "pushed into" says.
        relation = 'pushed into';
        level = effectiveLevel(relation, rawLevel);
      } else {
        // A pair dealt into the same category twice in one plan spreads across its
        // relations rather than asking for the same (relation, detail) again — that is the
        // intra-category variety the backfill exists for, and it saves the writer from
        // inventing two different positions for one combination in one batch.
        const cursor = cursors[keyOf(category.id, complexity)];
        for (let probe = 1; probe < cursor.relations.length && state.dealt.has(`${relation}|${level}`); probe++) {
          relation = cursor.relations[cursor.rel++ % cursor.relations.length];
          level = effectiveLevel(relation, rawLevel);
        }
      }
      if (level === 'bare') state.bare.add(relation);
      state.dealt.add(`${relation}|${level}`);

      assignments.push({ pair, category, complexity, relation, level, tier });
      remaining[keyOf(category.id, complexity)]--;
      state.total++;
      state.perCategory[category.id] = (state.perCategory[category.id] ?? 0) + 1;
      progress = true;
    }
  }
  return assignments;
}

/** Assignments into requests: one category per request so its prompt states one brief,
 *  chunked to BATCH, shuffled so categories interleave rather than run in blocks. */
function requestsOf(assignments) {
  const byCategory = new Map();
  for (const assignment of assignments) {
    const list = byCategory.get(assignment.category.id) ?? [];
    list.push(assignment);
    byCategory.set(assignment.category.id, list);
  }
  const requests = [];
  for (const list of byCategory.values()) {
    for (let i = 0; i < list.length; i += BATCH) requests.push(list.slice(i, i + BATCH));
  }
  return shuffled(requests);
}

// ---------------------------------------------------------------------------------- prompt

const systemFor = (category) => `You write placement phrases for a 3D placement dataset. Each numbered request gives a fixed pair of real assets — an anchor object and a placed object, each with a one-sentence description — plus an assigned spatial relation and a detail instruction. The pair is fixed: use both names exactly as given, never rename or swap them.

Every request in this batch uses the placement category "${category.label}": ${category.brief}

For every request return:
- i: the request's number.
- placement: the placed object's name, the assigned relation, the anchor's name — then whatever the request's detail line asks:
  - "no position detail": nothing more. "${category.examples.bare}"
  - "add one simple position": one plain position, your choice of where — "${category.examples.position}"
  - "tie it to a real part of the anchor": name one real part of the anchor asset and place the object against it, with a direction word if it helps — "${category.examples.part}"
- fit: your own judgment of this pair under this relation. "natural" when the scene reads like everyday life, "stretch" when it is physically workable but unusual, "nonsense" when the placed object's shape or material makes the relation impossible — rigid things cannot be laid over something, a larger thing cannot go inside a smaller one. Write the phrase for natural and stretch; for nonsense return an empty placement.

Each request lists the phrases this pair already has. The new phrase must put the placed object somewhere clearly different — a different spot, side, part or arrangement — never a rewording of one of them.

The placement phrase must read like plain everyday speech:
- write "the" before each object name after the first: "cup on top of the table", never "cup on top of table".
- simple common words only — no rare or formal verbs, no flourishes.
- neutral verbs only: never forceful or slangy words like "stabbed", "stuck", "jammed", "wedged" — say "inserted into", "attached to", "resting on" instead.
- the two names and the assigned relation appear word-for-word, with nothing added to the names.
- positions use plain direction words — top, bottom, middle, side, edge, corner, left, right, front, back. Never a person or viewpoint ("eye level", "arm's reach"), never the room ("near the wall").
- a part must be a real geometric feature of the anchor — a shelf, arm, leg, handle, roof, opening its description states or its type makes obvious. Never surface details like paint, logos or seams, which have no shape.
- you may add one everyday orientation or manner word where it reads naturally — sitting, lying on its side, folded, halfway in. Most phrases need none.
- across the batch, never repeat the same position wording, part, or manner word twice.`;

function userFor(batch) {
  return [
    `Write ${batch.length} placement phrase(s).`,
    '',
    ...batch.flatMap((assignment, i) => [
      `${i + 1}. ${assignment.relation} — ${DETAIL_LINES[assignment.level]}`,
      `   placed: ${assignment.pair.placed.name} — ${assignment.pair.placed.description}`,
      `   anchor: ${assignment.pair.anchor.name} — ${assignment.pair.anchor.description}`,
      `   already used: ${assignment.pair.existing.map((entry) => `"${entry.phrase}"`).join('; ')}`,
    ]),
  ].join('\n');
}

const SCHEMA = {
  type: 'object',
  properties: {
    items: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          i: { type: 'integer' },
          placement: { type: 'string' },
          fit: { type: 'string', enum: ['natural', 'stretch', 'nonsense'] },
        },
        required: ['i', 'placement', 'fit'],
        additionalProperties: false,
      },
    },
  },
  required: ['items'],
  additionalProperties: false,
};

// ------------------------------------------------------------------------------ validation

const ARTICLES = new Set(['the', 'a', 'an']);
// Words that dress a phrase without changing where the object ends up. Stripped before two
// phrases are compared, so "sitting on top in the middle" cannot pass as different from
// "on top in the middle", and "near the middle" cannot pass as different from "in the
// middle". Direction words stay — left against right is exactly the variety wanted.
const DRESSING = new Set([
  'sitting', 'resting', 'lying', 'placed', 'set', 'folded', 'halfway',
  'of', 'in', 'on', 'at', 'its', 'near', 'side',
]);

/** The words that actually locate the object: the phrase minus both names, the relation,
 *  articles and dressing. Two same-relation phrases whose remainders match are the same
 *  placement in different clothes. */
function remainder(phrase, pair, relation) {
  let list = words(phrase);
  for (const run of [pair.anchor.name, pair.placed.name, relation]) list = without(list, run);
  return new Set(list.filter((word) => !ARTICLES.has(word) && !DRESSING.has(word)));
}

const sameSet = (a, b) => a.size === b.size && [...a].every((word) => b.has(word));

/** Why a returned phrase is unusable, or null when it is fine. A nonsense verdict is the
 *  caller's to handle — it is a judgment about the pair, not about this phrase. */
function problemWith(phrase, { pair, relation }) {
  if (!phrase) return 'empty placement';
  if (words(phrase).length > PHRASE_MAX_WORDS) return `placement too long: "${phrase}"`;

  const lower = phrase.toLowerCase();
  if (!lower.includes(`the ${pair.anchor.name.toLowerCase()}`)) return `anchor not written as "the …": "${phrase}"`;
  if (!lower.includes(pair.placed.name.toLowerCase())) return `placement omits the placed object: "${phrase}"`;
  // The relation is what the sample's metadata claims, so the phrase has to actually say it.
  if (!lower.includes(relation)) return `assigned relation "${relation}" missing: "${phrase}"`;

  let rest = lower;
  for (const name of [pair.anchor.name, pair.placed.name]) rest = rest.replaceAll(name.toLowerCase(), ' ');
  const banned = rest.match(BANNED_VERBS);
  if (banned) return `banned word "${banned[0]}" in: "${phrase}"`;

  const norm = words(phrase).join(' ');
  const mine = remainder(phrase, pair, relation);
  for (const entry of pair.existing) {
    if (words(entry.phrase).join(' ') === norm) return `repeats "${entry.phrase}"`;
    if (entry.relation === relation && sameSet(remainder(entry.phrase, pair, relation), mine)) {
      return `same placement as "${entry.phrase}"`;
    }
  }
  return null;
}

// --------------------------------------------------------------------------------- samples

function writeSample(assignment, phrase, fit) {
  const { pair, category, complexity, relation, level } = assignment;
  const uuid = randomUUID();
  const id = `${meta.slug(pair.anchor.name)}_${uuid}`;
  const dir = path.join(GENERATED_DIR, id);
  fs.mkdirSync(dir, { recursive: true });

  const metadata = {
    id,
    uuid,
    created_at: new Date().toISOString(),
    context: pair.context,
    pair: pair.key,
    // natural only when the planner's tag test and the writer agree; either calling it a
    // stretch makes it one, so the field can be filtered on without re-deriving anything.
    tier: assignment.tier === 'natural' && fit === 'natural' ? 'natural' : 'stretch',
    category: category.id,
    complexity,
    relation,
    detail: level,
    placement: phrase,
    anchor: { name: pair.anchor.name, description: pair.anchor.description, objaverse: pair.anchor.objaverse },
    placed: { name: pair.placed.name, description: pair.placed.description, objaverse: pair.placed.objaverse },
  };

  meta.write(dir, metadata);
  meta.writeAtomic(path.join(dir, 'placement.txt'), meta.placementText(metadata));
  return metadata;
}

// ------------------------------------------------------------------------------------- run

function printEligibility(pairs, shortfalls) {
  const untagged = pairs.filter((pair) => !pair.tags.anchor || !pair.tags.placed).length;
  console.log(`  eligibility (pairs by tier):`);
  for (const category of CATEGORIES) {
    const natural = pairs.filter((pair) => pair.tier[category.id] === 'natural').length;
    const stretch = pairs.filter((pair) => pair.tier[category.id] === 'stretch').length;
    const short = (shortfalls[keyOf(category.id, 'simple')] ?? 0) + (shortfalls[keyOf(category.id, 'complex')] ?? 0);
    console.log(
      `    ${category.id.padEnd(12)} natural ${String(natural).padStart(6)}   stretch ${String(stretch).padStart(6)}` +
        `   shortfall ${String(short).padStart(6)}`,
    );
  }
  if (untagged) console.log(`    (${untagged} pair(s) missing pool tags — noncontact only for those)`);
}

export async function backfill({ relations, pairLimit = PAIR_LIMIT } = {}) {
  const tags = loadTags();
  const samples = (await meta.listAsync(GENERATED_DIR, { label: 'reading corpus' })).map((s) => s.metadata);
  const harvested = harvest(samples, tags);
  const { skipped } = harvested;
  let pairs = harvested.pairs;
  if (!pairs.length) throw new Error(`no seeded pairs in ${GENERATED_DIR}`);

  const corpusPairs = pairs.length;
  if (pairLimit > 0 && pairLimit < pairs.length) {
    pairs = [...pairs].sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0)).slice(0, pairLimit);
  }

  const created = {};
  for (const c of CATEGORIES) for (const cx of COMPLEXITIES) created[keyOf(c.id, cx)] = 0;
  for (const metadata of samples) {
    const key = keyOf(metadata.category, metadata.complexity === 'simple' ? 'simple' : 'complex');
    if (key in created) created[key]++;
  }

  // `relations` is how many backfilled samples the selected pairs should carry in total,
  // not how many to add. Absolute on purpose: a relative "add N more" recomputes a larger
  // goal every time it is invoked, so a crashed run that is simply re-issued overshoots,
  // and a later full run cannot count a finished smoke test as progress. Subtracting what
  // is already there makes re-running converge instead.
  const done = pairs.reduce((n, pair) => n + pair.backfills, 0);
  const budget = Math.max(0, relations - done);
  const target = samples.length + budget;
  const quotas = spread(target, CATEGORIES);
  const targets = { ...quotas }; // the pure weight split, kept for the final report
  const shortfallNow = () =>
    Object.fromEntries(Object.keys(quotas).map((key) => [key, Math.max(0, quotas[key] - created[key])]));

  console.log(
    `  ${pairs.length} pair(s)` +
      (pairs.length === corpusPairs ? '' : ` of ${corpusPairs} (first ${pairs.length} by sorted key)`) +
      ` across ${samples.length} sample(s)` +
      (skipped ? `, ${skipped} sample(s) without seeds or category skipped` : '') +
      ` — ${done}/${relations} backfilled relation(s) already, ${budget} to write`,
  );
  printEligibility(pairs, shortfallNow());

  if (DRY) {
    const plan = deal(shortfallNow(), pairs);
    const counts = {};
    for (const a of plan) {
      const key = `${a.category.id}/${a.complexity} ${a.tier}`;
      counts[key] = (counts[key] ?? 0) + 1;
    }
    console.log(`\n  plan: ${plan.length} assignment(s)`);
    for (const [key, n] of Object.entries(counts).sort()) console.log(`    ${key.padEnd(28)} ${n}`);

    const seen = new Set();
    for (const request of requestsOf(plan)) {
      if (seen.has(request[0].category.id)) continue;
      seen.add(request[0].category.id);
      console.log(`\n═══ ${request[0].category.id} ═══\n\n${systemFor(request[0].category)}\n\n--- user ---\n\n${userFor(request)}`);
    }
    return created;
  }

  const spend = { calls: 0, cost: 0 };
  const written = { natural: 0, stretch: 0 };
  const dropped = { nonsense: 0, problem: 0 };
  let overflowed = false;
  // The quota shortfall is not the same number as the budget: a corpus already under target
  // in some category carries that deficit into its shortfall, so the dealer would otherwise
  // write past what was asked for. This is the hard ceiling.
  let budgetLeft = budget;

  // Dropped and nonsense assignments leave a shortfall — re-plan it, a few rounds. Whatever
  // the dealt mix structurally cannot serve (soft, when too few pairs can drape) is moved
  // once, spread by weight across the categories that still have room, so the run lands on
  // its count with the mix bent as little as capacity allows.
  for (let round = 0; round < 8; round++) {
    if (budgetLeft <= 0) break;
    const short = shortfallNow();
    const residual = Object.values(short).reduce((n, v) => n + v, 0);
    if (!residual) break;

    let plan = deal(short, pairs);
    if (!overflowed && plan.length < residual) {
      overflowed = true;
      const stuck = residual - plan.length;
      const open = CATEGORIES.filter((category) =>
        pairs.some(
          (pair) =>
            pair.tier[category.id] &&
            pair.perCategory[category.id] < CATEGORY_CAP[category.id] &&
            pair.backfills < MAX_PER_PAIR,
        ),
      );
      if (open.length) {
        console.log(`  ${stuck} sample(s) cannot fill at the dealt mix — spread across ${open.map((c) => c.id).join(', ')} instead`);
        const plannedByKind = {};
        for (const a of plan) {
          const key = keyOf(a.category.id, a.complexity);
          plannedByKind[key] = (plannedByKind[key] ?? 0) + 1;
        }
        const extra = spread(stuck, open);
        for (const key of Object.keys(quotas)) {
          quotas[key] = (created[key] ?? 0) + (plannedByKind[key] ?? 0) + (extra[key] ?? 0);
        }
        plan = deal(shortfallNow(), pairs);
      }
    }
    if (plan.length > budgetLeft) plan = plan.slice(0, budgetLeft);
    if (!plan.length) break;

    const requests = requestsOf(plan);
    console.log(
      `\n  round ${round + 1}: ${plan.length} assignment(s) in ${requests.length} call(s), ${WIDTH} wide`,
    );
    // Per-sample lines below are the point of a smoke test, so the tracker prints whole
    // lines on an interval rather than a bar that they would trample.
    const progress = track('    writing', plan.length, { indent: '' });

    await mapLimit(requests, WIDTH, async (batch) => {
      const category = batch[0].category;
      let items;
      try {
        const reply = await retry(() =>
          chatJSON({
            model: MODEL,
            system: systemFor(category),
            user: userFor(batch),
            name: 'backfill',
            schema: SCHEMA,
            temperature: 1,
            reasoning: REASONING,
          }),
        );
        items = new Map((reply.data.items ?? []).map((item) => [item.i, item]));
        spend.calls++;
        if (typeof reply.usage?.cost === 'number') spend.cost += reply.usage.cost;
      } catch (err) {
        console.error(`  ✗ [${category.id}] batch of ${batch.length}: ${err.message}`);
        progress.tick(batch.length);
        return;
      }

      // Nothing awaits below, so validation reads and folder writes cannot interleave with
      // another batch's — the per-pair phrase lists stay trustworthy.
      for (const [i, assignment] of batch.entries()) {
        const { pair, complexity, relation, level } = assignment;
        const kind = keyOf(category.id, complexity);
        if (budgetLeft <= 0) continue;
        if (created[kind] >= quotas[kind]) continue;
        if (pair.backfills >= MAX_PER_PAIR || pair.perCategory[category.id] >= CATEGORY_CAP[category.id]) continue;

        const item = items.get(i + 1);
        if (!item) {
          dropped.problem++;
          console.error(`  ✗ [${category.id}] ${pair.placed.name} × ${pair.anchor.name}: dropped by the model`);
          continue;
        }
        if (item.fit === 'nonsense') {
          // The writer looked at the actual pair and refused the category outright. Retire
          // the combination so re-planning does not deal it again this run.
          pair.tier[category.id] = null;
          dropped.nonsense++;
          console.error(`  ✗ [${category.id}] ${pair.placed.name} × ${pair.anchor.name}: nonsense — retired`);
          continue;
        }

        const phrase = (item.placement ?? '').trim();
        const problem = problemWith(phrase, assignment);
        if (problem) {
          dropped.problem++;
          console.error(`  ✗ [${category.id}] dropped: ${problem}`);
          continue;
        }

        const metadata = writeSample(assignment, phrase, item.fit);
        pair.existing.push({ category: category.id, relation, detail: level, phrase });
        pair.backfills++;
        pair.perCategory[category.id]++;
        created[kind]++;
        written[metadata.tier]++;
        budgetLeft--;
        console.log(`  + [${category.id}/${complexity}·${level}/${metadata.tier}] ${phrase}`);
      }

      progress.note(
        `$${spend.cost.toFixed(4)} · ${written.natural + written.stretch} written` +
          (dropped.nonsense + dropped.problem ? ` · ${dropped.nonsense + dropped.problem} dropped` : ''),
      );
      progress.tick(batch.length);
    });
    progress.done();
  }

  const made = written.natural + written.stretch;
  console.log(
    `\n  ${spend.calls} call(s) to ${MODEL}, $${spend.cost.toFixed(4)} — ${made} written` +
      ` (${written.natural} natural, ${written.stretch} stretch), ${dropped.nonsense} nonsense, ${dropped.problem} dropped`,
  );

  // The corpus against the pure weight split, and how the pairs came out.
  console.log(`\n  corpus by category (target in parentheses):`);
  for (const category of CATEGORIES) {
    const have = created[keyOf(category.id, 'simple')] + created[keyOf(category.id, 'complex')];
    const want = targets[keyOf(category.id, 'simple')] + targets[keyOf(category.id, 'complex')];
    console.log(`    ${category.id.padEnd(12)} ${String(have).padStart(6)}  (${want})`);
  }
  const histogram = {};
  let covered = 0;
  for (const pair of pairs) {
    histogram[pair.backfills] = (histogram[pair.backfills] ?? 0) + 1;
    if (new Set(pair.existing.map((entry) => entry.category)).size >= 2) covered++;
  }
  console.log(
    `  backfills per pair: ${Object.entries(histogram).sort(([a], [b]) => a - b).map(([n, c]) => `${n}×${c}`).join('  ')}` +
      `\n  ${covered}/${pairs.length} pair(s) now span ≥2 categories`,
  );

  if (budgetLeft > 0) {
    console.log(
      `  ${budgetLeft} of the ${budget} asked for could not be dealt — capacity, not budget.` +
        ' Re-running continues from here.',
    );
  }
  return created;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const relations = Number(flag('relations', NaN));
  if (!Number.isInteger(relations) || relations <= 0) {
    console.error(
      'usage: node pipeline/backfill-relations.mjs --relations=N [--pairs=N] [--dry] [--out=dir]\n' +
        '  --relations is the total backfilled relations wanted, not how many to add, so\n' +
        '  re-running the same command resumes rather than overshooting.',
    );
    process.exit(2);
  }

  console.log(`\n1b · backfilling relationships across existing pairs via ${MODEL}`);
  await backfill({ relations });
  if (!DRY) console.log('\nMeshes come next: the objaverse fetch stage reads each sample\'s recorded uid.');
}
