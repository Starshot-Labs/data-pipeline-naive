// Stage 1 for the retrieval pipeline: invent placement scenes seeded with real Objaverse
// assets, dealt across the six placement categories by weight — rigid counts double and
// soft one-and-a-half, so a 50k run comes out ~13.3k rigid, 10k soft, ~6.7k of each rest.
//
//   node pipeline/generate-scenes.mjs --samples=50000    generate up to 50k scenes
//   node pipeline/generate-scenes.mjs --samples=12 --dry print one prompt per batch kind
//
// The code deals only abstract knobs; the model binds them to the pair it chose. Relations
// are pure spatial primitives ("on top of", "inside"), and each sample also gets a dealt
// detail level: bare (names and relation only), position (the model adds one plain
// direction of its choosing), or part (the model ties the placement to a real geometric
// part of the chosen anchor, read off its caption). The simple half of each category deals
// bare/position, the complex half position/part — which side, which part, how deep, what
// orientation is always the model's per-pairing decision.
//
// Diversity is dealt, not hoped for. The plan fixes every axis before any call is made —
// category quota, simple/complex half, detail level, context (shuffled and dealt within
// each category), and relation (dealt round-robin inside each request) — so 6k penetrative
// samples cannot collapse into 6k knives in cutting boards.
//
// Every request offers real assets from the pool built by objaverse-pool.mjs. The anchor
// MUST be one of them, which is the point of seeding: the sample records the asset's uid
// and GLB path at birth, so there is no retrieval-matching step to miss later, and the
// anchor's description is written faithful to a mesh that actually exists. Placed objects
// prefer a seed but may be invented when nothing offered fits — those fall back to
// embedding retrieval downstream, and record no uid.
//
// Re-running is safe: existing samples count toward each category's quota, duplicates are
// dropped against everything already on disk, and each sample folder is written the moment
// its batch lands.

import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chatJSON } from './openrouter.mjs';
import { mapLimit, retry, widthOf } from './limit.mjs';
import { CATEGORIES, CONTEXTS } from './scene-spec.mjs';
import * as meta from './metadata.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ENV_FILE = path.join(ROOT, '.env');
if (fs.existsSync(ENV_FILE)) process.loadEnvFile(ENV_FILE);

const args = process.argv.slice(2);
const flag = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

const GENERATED_DIR = path.resolve(ROOT, flag('out', process.env.GENERATED_DIR ?? 'generated'));
const POOL_FILE = path.join(path.resolve(ROOT, flag('pool', process.env.POOL_DIR ?? 'pool')), 'pool.jsonl');
const DRY = args.includes('--dry');
const TARGET = Number(flag('samples', args.find((a) => !a.startsWith('--')) ?? 0));

const MODEL = process.env.SPEC_MODEL ?? 'google/gemini-3.6-flash';
// Generation is pattern-filling, not puzzle-solving — a flash-class model handles it, and
// thinking only adds output tokens. Reasoning cannot be turned off on gemini-3.6-flash
// (OpenRouter answers 400), so the default is the lowest effort it allows;
// SPEC_REASONING=off works only on models that permit disabling.
const REASONING = (() => {
  const value = (process.env.SPEC_REASONING ?? 'low').trim().toLowerCase();
  return value === 'off' ? { enabled: false } : { effort: value };
})();
const BATCH = Number(process.env.SPEC_BATCH ?? 20);
const WIDTH = widthOf('SPEC_CONCURRENCY');
// Exact-match license allow-list, e.g. POOL_LICENSES=by,by-sa,cc0. Unset allows everything.
const LICENSES = (process.env.POOL_LICENSES ?? '').split(',').map((s) => s.trim()).filter(Boolean);

const PHRASE_MAX_WORDS = 15;
const words = (text) => text.trim().split(/\s+/).filter(Boolean);

// Forceful or slangy verbs a phrase may not use. Checked with the object names stripped
// out first, so a "hockey stick" or "tent stake" never trips over its own name.
const BANNED_VERBS = /\b(stab|stuck|stick|jam|shov|wedg|cram|smash|slam|stak|pierc|embed)\w*/i;

// Decorative words a multi-word name may not carry — the name is what a thing is, not what
// it looks like. Single-word names are exempt so the fruit "orange" stays legal.
const COLOR_WORDS = new Set([
  'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'black', 'white', 'brown',
  'gray', 'grey', 'golden', 'gold', 'silver', 'colorful', 'colourful', 'striped', 'spotted',
  'dark', 'pale', 'bright', 'shiny', 'rusty', 'vintage', 'stylized', 'fancy',
]);

function shuffled(items) {
  const deck = [...items];
  for (let i = deck.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [deck[i], deck[j]] = [deck[j], deck[i]];
  }
  return deck;
}

// ---------------------------------------------------------------------------------- pool

/** Walks a shuffled deck with a cursor, reshuffling on wrap, so asset usage across the run
 *  stays near-uniform and one take never repeats a uid. */
function sampler(items) {
  let deck = shuffled(items);
  let cursor = 0;
  return (k) => {
    const take = new Map();
    const want = Math.min(k, items.length);
    while (take.size < want) {
      if (cursor >= deck.length) {
        deck = shuffled(deck);
        cursor = 0;
      }
      const item = deck[cursor++];
      take.set(item.uid, item);
    }
    return [...take.values()];
  };
}

function loadPool() {
  if (!fs.existsSync(POOL_FILE)) {
    throw new Error(`${POOL_FILE} not found — build it first: node pipeline/objaverse-pool.mjs`);
  }
  const rows = [];
  for (const line of fs.readFileSync(POOL_FILE, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    try { rows.push(JSON.parse(line)); } catch { /* a line cut off mid-append */ }
  }
  const usable = rows.filter(
    (row) => !row.junk && (!LICENSES.length || LICENSES.includes(row.license ?? 'unknown')),
  );

  const buckets = {};
  for (const category of CATEGORIES) {
    // Hovering above something needs no affordance from it, so every usable asset anchors
    // noncontact regardless of how the tagger read its caption.
    const anchors =
      category.id === 'noncontact' ? usable : usable.filter((row) => row.anchor?.includes(category.id));
    const placed = usable.filter((row) => row.placed?.includes(category.id));
    buckets[category.id] = {
      size: anchors.length,
      anchors: sampler(anchors),
      placed: sampler(placed),
      placedSize: placed.length,
    };
  }
  console.log(`  pool: ${usable.length}/${rows.length} usable asset(s) from ${POOL_FILE}`);
  return buckets;
}

// ---------------------------------------------------------------------------------- plan

const COMPLEXITIES = ['simple', 'complex'];
const keyOf = (categoryId, complexity) => `${categoryId}|${complexity}`;

/**
 * Split of `total` across the categories in proportion to their weights, the fractional
 * remainder going to the categories closest to their next whole sample. Each category's
 * quota is then halved between the simple and complex phrase styles, simple taking the odd
 * one.
 */
function quotasFor(total) {
  const totalWeight = CATEGORIES.reduce((sum, c) => sum + c.weight, 0);
  const shares = CATEGORIES.map((c) => ({ id: c.id, exact: (total * c.weight) / totalWeight }));
  const perCategory = Object.fromEntries(shares.map((s) => [s.id, Math.floor(s.exact)]));
  let left = total - Object.values(perCategory).reduce((sum, q) => sum + q, 0);
  for (const { id } of [...shares].sort((a, b) => (b.exact % 1) - (a.exact % 1))) {
    if (left-- <= 0) break;
    perCategory[id]++;
  }

  const quotas = {};
  for (const category of CATEGORIES) {
    const quota = perCategory[category.id];
    quotas[keyOf(category.id, 'simple')] = Math.ceil(quota / 2);
    quotas[keyOf(category.id, 'complex')] = Math.floor(quota / 2);
  }
  return quotas;
}

/**
 * Requests for every (category, phrase style) shortfall: contexts shuffled and dealt within
 * the category, relations dealt round-robin across its samples with a random starting
 * offset, at most `BATCH` samples (and never more than the anchor bucket holds) per
 * request. Every request is entirely simple or entirely complex, so its prompt states one
 * phrase shape rather than two. The finished plan is shuffled so categories interleave
 * rather than run in blocks.
 */
function planRequests(shortfalls, buckets) {
  const plan = [];
  for (const category of CATEGORIES) {
    const bucket = buckets[category.id];
    if (!bucket.size) {
      console.error(`  ✗ ${category.id}: no anchors in the pool — tag more assets first`);
      continue;
    }

    for (const complexity of COMPLEXITIES) {
      let left = shortfalls[keyOf(category.id, complexity)] ?? 0;
      if (left <= 0) continue;

      const relations = category[complexity].relations;
      // The dealt detail levels: how much the phrase says beyond names + relation. The
      // simple half alternates nothing at all with one plain direction; the complex half
      // alternates a direction with a real part of the anchor. Content is the model's.
      const levels = complexity === 'simple' ? ['bare', 'position'] : ['position', 'part'];
      const contexts = shuffled(CONTEXTS);
      const width = Math.min(BATCH, bucket.size);
      const offset = Math.floor(Math.random() * relations.length);
      const levelOffset = Math.floor(Math.random() * levels.length);
      let slot = 0;
      let levelSlot = 0;
      let turn = 0;
      while (left > 0) {
        const count = Math.min(left, width);
        plan.push({
          category,
          complexity,
          context: contexts[turn++ % contexts.length],
          relations: Array.from({ length: count }, () => relations[(offset + slot++) % relations.length]),
          levels: Array.from({ length: count }, () => levels[(levelOffset + levelSlot++) % levels.length]),
        });
        left -= count;
      }
    }
  }
  return shuffled(plan);
}

// -------------------------------------------------------------------------------- prompt

const systemFor = (category, complexity) => `You compose samples for a 3D placement dataset. Each sample pairs an anchor — a real 3D asset chosen from the request's list — with a placed object, plus one brief phrase stating exactly where the placed object goes relative to the anchor.

Every sample in this batch uses the placement category "${category.label}": ${category.brief}
The anchor must afford: ${category.anchor}.
The placed object must be: ${category.placed}.

For every sample return:
- anchor_uid: the uid of the chosen anchor asset — a different asset for every sample, picked because its geometry genuinely affords the sample's assigned relation.
- anchor_name: the plain everyday name of the thing, one or two lowercase words where possible — "bird", never "colorful bird"; a material word only when it is part of the common name ("cutting board").
- anchor_description: one sentence of the anchor's visible form, material, colour and proportions, faithful to its caption. Describe the object alone — never a room, a background, lighting, or another object.
- placed_uid: the uid of a provided placed-object asset when one fits the relation, otherwise null.
- placed_name / placed_description: same rules. An invented placed object must still satisfy the category's requirement above.
- placement: the placed object's name, the sample's assigned relation, the anchor's name — then whatever the sample's detail line asks:
  - "no position detail": nothing more. "${category.examples.bare}"
  - "add one simple position": one plain position, your choice of where — "${category.examples.position}"
  - "tie it to a real part of the anchor": name one real part of the chosen asset and place the object against it, with a direction word if it helps — "${category.examples.part}"

The placement phrase must read like plain everyday speech:
- write "the" before each object name after the first: "cup on top of the table", never "cup on top of table".
- simple common words only — no rare or formal verbs, no flourishes.
- neutral verbs only: never forceful or slangy words like "stabbed", "stuck", "jammed", "wedged" — say "inserted into", "attached to", "resting on" instead.
- the two names appear word-for-word with nothing added to them: "knife inserted into the tree trunk", never "knife inserted into the tall old tree trunk".
- positions use plain direction words — top, bottom, middle, side, edge, corner, left, right, front, back. Never a person or viewpoint ("eye level", "arm's reach"), never the room ("near the wall").
- a part must be a real geometric feature of the chosen asset — a shelf, arm, leg, handle, roof, opening its caption states or its type makes obvious. Never surface details like paint, logos or seams, which have no shape.
- you may add one everyday orientation or manner word where it reads naturally — sitting, lying on its side, folded, halfway in. Most samples need none.
- across the batch, never repeat the same position wording, part, or manner word twice.

Rules:
- names are generic: no brand names, no proper nouns.
- every anchor/placed pair in the batch must be distinct and plausible in the given setting.`;

const assetLine = (asset) => `- ${asset.uid} — ${asset.caption.slice(0, 220)}`;

const LEVEL_LINE = {
  bare: 'no position detail',
  position: 'add one simple position',
  part: 'tie it to a real part of the anchor',
};

function userFor({ context, relations, levels }, anchors, placed, avoid) {
  return [
    `Compose ${relations.length} sample(s) set in a ${context}.`,
    '',
    'Assigned relations, one per sample in order:',
    ...relations.map((relation, i) => `${i + 1}. ${relation} — ${LEVEL_LINE[levels[i]]}`),
    '',
    'Anchor assets — choose a different one for each sample:',
    ...anchors.map(assetLine),
    ...(placed.length
      ? ['', 'Placed-object assets — use one when it fits, otherwise invent (placed_uid null):', ...placed.map(assetLine)]
      : ['', 'No placed-object assets are offered — invent every placed object (placed_uid null).']),
    ...(avoid.length ? ['', `Avoid these placed objects, already used in this setting: ${avoid.join(', ')}`] : []),
  ].join('\n');
}

const SCHEMA = {
  type: 'object',
  properties: {
    samples: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          anchor_uid: { type: 'string' },
          anchor_name: { type: 'string' },
          anchor_description: { type: 'string' },
          placed_uid: { type: ['string', 'null'] },
          placed_name: { type: 'string' },
          placed_description: { type: 'string' },
          placement: { type: 'string' },
        },
        required: [
          'anchor_uid', 'anchor_name', 'anchor_description',
          'placed_uid', 'placed_name', 'placed_description', 'placement',
        ],
        additionalProperties: false,
      },
    },
  },
  required: ['samples'],
  additionalProperties: false,
};

// ------------------------------------------------------------------------------ samples

/** Why a returned sample is unusable, or null when it is fine. */
function problemWith(sample, offered, usedAnchors) {
  const anchor = offered.anchors.get(sample.anchor_uid);
  if (!anchor) return `anchor_uid ${sample.anchor_uid} was not offered`;
  if (usedAnchors.has(sample.anchor_uid)) return 'anchor asset reused within the batch';
  if (sample.placed_uid && !offered.placed.has(sample.placed_uid)) return `placed_uid ${sample.placed_uid} was not offered`;

  for (const name of [sample.anchor_name, sample.placed_name]) {
    const parts = words(name ?? '');
    if (!parts.length || parts.length > 3) return `bad name "${name}"`;
    // "bird", never "colorful bird" — but a name that IS a colour ("orange") survives.
    if (parts.length > 1 && parts.some((word) => COLOR_WORDS.has(word.toLowerCase()))) {
      return `name carries a colour or style word: "${name}"`;
    }
  }
  if (!sample.anchor_description?.trim() || !sample.placed_description?.trim()) return 'empty description';

  const phrase = sample.placement?.trim() ?? '';
  if (words(phrase).length > PHRASE_MAX_WORDS) return `placement too long: "${phrase}"`;
  const lower = phrase.toLowerCase();
  // The anchor always reads as "the <name>" — the phrase leads with the placed object, so
  // a bare anchor name is the model dropping an article, which reads broken.
  if (!lower.includes(`the ${sample.anchor_name.toLowerCase()}`)) return `anchor not written as "the …": "${phrase}"`;
  if (!lower.includes(sample.placed_name.toLowerCase())) return `placement omits the placed object: "${phrase}"`;

  let rest = lower;
  for (const name of [sample.anchor_name, sample.placed_name]) rest = rest.replaceAll(name.toLowerCase(), ' ');
  const banned = rest.match(BANNED_VERBS);
  if (banned) return `banned word "${banned[0]}" in: "${phrase}"`;
  return null;
}

const seedOf = (asset) => ({
  uid: asset.uid,
  glb: asset.glb,
  ...(asset.license ? { license: asset.license } : {}),
});

function writeSample(sample, request, relation, level, offered) {
  const uuid = randomUUID();
  const id = `${meta.slug(sample.anchor_name)}_${uuid}`;
  const dir = path.join(GENERATED_DIR, id);
  fs.mkdirSync(dir, { recursive: true });

  const placedSeed = sample.placed_uid ? offered.placed.get(sample.placed_uid) : null;
  const metadata = {
    id,
    uuid,
    created_at: new Date().toISOString(),
    context: request.context,
    category: request.category.id,
    complexity: request.complexity,
    relation,
    detail: level,
    placement: sample.placement.trim(),
    anchor: {
      name: sample.anchor_name.trim(),
      description: sample.anchor_description.trim(),
      objaverse: seedOf(offered.anchors.get(sample.anchor_uid)),
    },
    placed: {
      name: sample.placed_name.trim(),
      description: sample.placed_description.trim(),
      ...(placedSeed ? { objaverse: seedOf(placedSeed) } : {}),
    },
  };

  meta.write(dir, metadata);
  meta.writeAtomic(path.join(dir, 'placement.txt'), metadata.placement);
  return metadata;
}

/** Dedup keys for a sample, textual and by seed, checked against everything on disk. */
function keysOf({ category, anchorUid, anchorName, placedName, relation, placement }) {
  return [
    `${category}|${anchorUid}|${meta.slug(placedName)}|${relation}`,
    `${meta.slug(anchorName)}|${meta.slug(placedName)}|${placement.trim().toLowerCase()}`,
  ];
}

async function generate(target) {
  fs.mkdirSync(GENERATED_DIR, { recursive: true });
  const buckets = loadPool();

  // Existing samples count toward quotas and seed the dedup set, so re-running only fills.
  // A sample from before the styles were split carries no `complexity` and counts as
  // complex, which is the style it was written in.
  const existing = meta.list(GENERATED_DIR).map((sample) => sample.metadata);
  const created = {};
  for (const c of CATEGORIES) for (const cx of COMPLEXITIES) created[keyOf(c.id, cx)] = 0;
  for (const m of existing) {
    const key = keyOf(m.category, m.complexity === 'simple' ? 'simple' : 'complex');
    if (key in created) created[key]++;
  }

  const seen = new Set();
  const usedPlaced = new Map(); // `${category}|${context}` → recent placed names
  for (const m of existing) {
    if (!m.category) continue;
    for (const key of keysOf({
      category: m.category,
      anchorUid: m.anchor.objaverse?.uid ?? '',
      anchorName: m.anchor.name,
      placedName: m.placed.name,
      relation: m.relation ?? '',
      placement: m.placement ?? '',
    })) seen.add(key);
  }

  const quotas = quotasFor(target);
  const spend = { calls: 0, cost: 0 };
  const shortfallNow = () =>
    Object.fromEntries(Object.keys(quotas).map((key) => [key, Math.max(0, quotas[key] - created[key])]));

  if (DRY) {
    const firstPerKind = new Map();
    for (const request of planRequests(shortfallNow(), buckets)) {
      const kind = keyOf(request.category.id, request.complexity);
      if (!firstPerKind.has(kind)) firstPerKind.set(kind, request);
    }
    for (const [kind, request] of firstPerKind) {
      const bucket = buckets[request.category.id];
      const anchors = bucket.anchors(Math.min(2 * request.relations.length, bucket.size));
      const placed = bucket.placed(Math.min(30, bucket.placedSize));
      console.log(`\n═══ ${kind} ═══\n\n${systemFor(request.category, request.complexity)}\n\n--- user ---\n\n${userFor(request, anchors, placed, [])}`);
    }
    return created;
  }

  // Dropped duplicates and invalid samples leave a shortfall — re-plan it, a few rounds.
  for (let round = 0; round < 6; round++) {
    const plan = planRequests(shortfallNow(), buckets);
    if (!plan.length) break;
    if (round) console.log(`  round ${round + 1}: ${plan.reduce((n, r) => n + r.relations.length, 0)} sample(s) still short`);

    await mapLimit(plan, WIDTH, async (request) => {
      const { category, complexity, context, relations } = request;
      const kind = keyOf(category.id, complexity);
      const bucket = buckets[category.id];
      const anchors = bucket.anchors(Math.min(2 * relations.length, bucket.size));
      const placed = bucket.placed(Math.min(30, bucket.placedSize));
      const offered = {
        anchors: new Map(anchors.map((a) => [a.uid, a])),
        placed: new Map(placed.map((p) => [p.uid, p])),
      };
      const cell = `${category.id}|${context}`;
      const avoid = (usedPlaced.get(cell) ?? []).slice(-40);

      let samples;
      try {
        const reply = await retry(() =>
          chatJSON({
            model: MODEL,
            system: systemFor(category, complexity),
            user: userFor(request, anchors, placed, avoid),
            name: 'scenes',
            schema: SCHEMA,
            temperature: 1,
            reasoning: REASONING,
          }),
        );
        samples = reply.data.samples;
        spend.calls++;
        if (typeof reply.usage?.cost === 'number') spend.cost += reply.usage.cost;
      } catch (err) {
        console.error(`  ✗ [${category.id}/${context}] ${err.message}`);
        return;
      }

      // Same convention as build.mjs: nothing awaits below, so dedup checks and folder
      // writes cannot interleave with another worker's.
      const usedAnchors = new Set();
      for (const [i, sample] of (samples ?? []).slice(0, relations.length).entries()) {
        if (created[kind] >= quotas[kind]) return;
        const relation = relations[i];
        const level = request.levels[i];

        const problem = problemWith(sample, offered, usedAnchors);
        if (problem) {
          console.error(`  ✗ [${category.id}/${context}] dropped: ${problem}`);
          continue;
        }
        const keys = keysOf({
          category: category.id,
          anchorUid: sample.anchor_uid,
          anchorName: sample.anchor_name,
          placedName: sample.placed_name,
          relation,
          placement: sample.placement,
        });
        if (keys.some((key) => seen.has(key))) continue;

        for (const key of keys) seen.add(key);
        usedAnchors.add(sample.anchor_uid);
        usedPlaced.set(cell, [...(usedPlaced.get(cell) ?? []), sample.placed_name.toLowerCase()]);
        writeSample(sample, request, relation, level, offered);
        created[kind]++;
        console.log(`  + [${category.id}/${complexity}·${level}/${context}] ${sample.placement.trim()}`);
      }
    });
  }
  if (spend.calls) console.log(`\n  ${spend.calls} call(s) to ${MODEL}, $${spend.cost.toFixed(4)}`);
  return created;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (!Number.isInteger(TARGET) || TARGET < 0) {
    console.error('usage: node pipeline/generate-scenes.mjs --samples=N [--dry] [--out=dir]');
    process.exit(2);
  }

  console.log(`\n1 · generating ${TARGET} scene(s) across ${CATEGORIES.length} categories via ${MODEL}`);
  const created = await generate(TARGET);

  if (!DRY) {
    const counts = Object.fromEntries(CATEGORIES.map((c) => [c.id, { simple: 0, complex: 0 }]));
    for (const sample of meta.list(GENERATED_DIR)) {
      const per = counts[sample.metadata.category];
      if (per) per[sample.metadata.complexity === 'simple' ? 'simple' : 'complex']++;
    }
    const total = Object.values(counts).reduce((n, c) => n + c.simple + c.complex, 0);
    console.log(`\n${total} categorized sample(s) in ${GENERATED_DIR}:`);
    for (const category of CATEGORIES) {
      const { simple, complex } = counts[category.id];
      console.log(`  ${category.id.padEnd(12)} ${String(simple + complex).padStart(5)}   (${simple} simple / ${complex} complex)`);
    }
    console.log("\nMeshes come next: the objaverse fetch stage reads each sample's recorded uid.");
  }
}
