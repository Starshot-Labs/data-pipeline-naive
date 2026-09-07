// A readable audit of what a backfill run actually produced.
//
//   node scripts/backfill-report.mjs                          the default corpus
//   node scripts/backfill-report.mjs --source=<dir> --out=report.md
//   node scripts/backfill-report.mjs --limit-examples=0       every phrase, not a sample
//
// The point is manual verification: the phrases are grouped under the pair they belong to,
// with the pair's original placement shown first, because a new phrase is only judgeable
// against the one it is supposed to differ from. Everything else in the file exists to
// answer "did the run do what it claimed" without opening the corpus — quota accuracy,
// per-pair variety, how far each sample got down the pipeline, what physics made of the
// poses, and a re-run of the writer's own validation rules against what landed on disk.
//
// Nothing here mutates the corpus, so it is safe to run against a live one mid-flight.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { mapLimit, widthOf } from '../pipeline/limit.mjs';
import { CATEGORIES } from '../pipeline/scene-spec.mjs';
import { track } from '../pipeline/progress.mjs';
import * as meta from '../pipeline/metadata.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ENV_FILE = path.join(ROOT, '.env');
if (fs.existsSync(ENV_FILE)) process.loadEnvFile(ENV_FILE);

const args = process.argv.slice(2);
const flag = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

const SOURCE_DIR = path.resolve(ROOT, flag('source', process.env.GENERATED_DIR ?? 'generated'));
const SCENE = process.env.SCENE_DIR ?? '/scene';
const PUBLISH_DIR = flag('published', path.join(SCENE, process.env.SCENE_PUBLISH_PREFIX ?? 'datasets/raw/stage1'));
const OUT = path.resolve(ROOT, flag('out', 'backfill-report.md'));
// 0 prints every pair. The default keeps a first look readable while still being a large
// enough sample to judge the writer on.
const LIMIT_EXAMPLES = Number(flag('limit-examples', 0));
const WIDTH = widthOf('REPORT_CONCURRENCY', 64);

const PHRASE_MAX_WORDS = 15;
const BANNED_VERBS = /\b(stab|stuck|stick|jam|shov|wedg|cram|smash|slam|stak|pierc|embed)\w*/i;
const words = (text) => text.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, ' ').split(/\s+/).filter(Boolean);

const pct = (n, d) => (d ? `${((n / d) * 100).toFixed(1)}%` : '—');
const pad = (v, n) => String(v).padStart(n);

/** Every sample under `root`, read concurrently — a serial walk over tens of thousands of
 *  these off a network volume is minutes of nothing happening. */
async function readCorpus(root) {
  if (!fs.existsSync(root)) throw new Error(`${root} does not exist`);
  const ids = fs
    .readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();

  const progress = track('reading corpus', ids.length);
  const samples = await mapLimit(ids, WIDTH, async (id) => {
    try {
      const raw = await fs.promises.readFile(path.join(root, id, meta.FILE), 'utf8');
      return { id, metadata: JSON.parse(raw) };
    } catch (err) {
      if (err.code === 'ENOENT') return null; // a folder mid-write, not a sample
      return { id, broken: err.message };
    } finally {
      progress.tick();
    }
  });
  progress.done();
  return samples.filter(Boolean);
}

/** Sample ids already carrying a metadata.json in the publish prefix, when it is reachable.
 *  Absent locally, which is fine — the report just omits the publish column then. */
async function readPublished(dir) {
  if (!fs.existsSync(dir)) return null;
  const ids = fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name);
  const progress = track('checking published', ids.length);
  const present = await mapLimit(ids, WIDTH, async (id) => {
    const found = await fs.promises.access(path.join(dir, id, meta.FILE)).then(() => id, () => null);
    progress.tick();
    return found;
  });
  progress.done();
  return new Set(present.filter(Boolean));
}

/** The writer's own rules, re-run against what is actually on disk. A phrase that fails
 *  here got past validation and into the corpus, which is a bug worth seeing. */
function recheck(sample, pair) {
  const phrase = meta.phraseOf(sample);
  const anchor = sample.anchor.name.toLowerCase();
  const placed = sample.placed.name.toLowerCase();
  const lower = phrase.toLowerCase();
  const problems = [];

  if (!phrase) problems.push('empty phrase');
  if (words(phrase).length > PHRASE_MAX_WORDS) problems.push(`over ${PHRASE_MAX_WORDS} words`);
  if (!lower.includes(`the ${anchor}`)) problems.push('anchor not written as "the …"');
  if (!lower.includes(placed)) problems.push('placed object not named');
  if (sample.relation && !lower.includes(sample.relation)) problems.push(`relation "${sample.relation}" absent`);

  let rest = lower;
  for (const name of [anchor, placed]) rest = rest.replaceAll(name, ' ');
  const banned = rest.match(BANNED_VERBS);
  if (banned) problems.push(`banned verb "${banned[0]}"`);

  // Novelty is a property of the pair, not the sample: an exact repeat of any other phrase
  // the same pair carries is the failure this whole exercise exists to prevent.
  const norm = words(phrase).join(' ');
  if (pair && pair.phrases.filter((other) => words(other).join(' ') === norm).length > 1) {
    problems.push('exact duplicate within the pair');
  }
  return problems;
}

const tally = (rows, key) => {
  const counts = {};
  for (const row of rows) counts[key(row)] = (counts[key(row)] ?? 0) + 1;
  return counts;
};

const table = (headers, rows) =>
  [
    `| ${headers.join(' | ')} |`,
    `| ${headers.map(() => '---').join(' | ')} |`,
    ...rows.map((row) => `| ${row.join(' | ')} |`),
  ].join('\n');

async function build() {
  const all = await readCorpus(SOURCE_DIR);
  const broken = all.filter((s) => s.broken);
  const samples = all.filter((s) => !s.broken).map((s) => s.metadata);
  const published = await readPublished(PUBLISH_DIR);

  const backfilled = samples.filter((s) => s.pair);
  const original = samples.filter((s) => !s.pair);

  // Pair index over the whole corpus, so a backfill's phrases can be shown against the
  // originals they had to differ from.
  const pairs = new Map();
  for (const sample of samples) {
    const key = sample.pair ?? `${sample.anchor?.objaverse?.uid}|${sample.placed?.objaverse?.uid}`;
    let pair = pairs.get(key);
    if (!pair) {
      pair = { key, anchor: sample.anchor.name, placed: sample.placed.name, original: [], added: [], phrases: [] };
      pairs.set(key, pair);
    }
    (sample.pair ? pair.added : pair.original).push(sample);
    pair.phrases.push(meta.phraseOf(sample));
  }
  const touched = [...pairs.values()].filter((pair) => pair.added.length);

  // ------------------------------------------------------------------ pipeline completion
  const stage = { written: backfilled.length, fetched: 0, placed: 0, published: 0, physics: 0 };
  for (const sample of backfilled) {
    if (meta.ROLES.every((role) => sample[role]?.mesh)) stage.fetched++;
    if (Array.isArray(sample.combined_size)) stage.placed++;
    if (sample.physics) stage.physics++;
    if (published?.has(sample.id)) stage.published++;
  }

  // ------------------------------------------------------------------------ quota accuracy
  const weightTotal = CATEGORIES.reduce((sum, c) => sum + c.weight, 0);
  const byCategory = tally(samples, (s) => s.category);
  const addedByCategory = tally(backfilled, (s) => s.category);
  const categoryRows = CATEGORIES.map((category) => {
    const have = byCategory[category.id] ?? 0;
    const want = Math.round((samples.length * category.weight) / weightTotal);
    const drift = have - want;
    return [
      `\`${category.id}\``,
      pad(addedByCategory[category.id] ?? 0, 5),
      pad(have, 6),
      pad(want, 6),
      `${drift >= 0 ? '+' : ''}${drift}`,
      pct(have, samples.length),
    ];
  });

  // ------------------------------------------------------------------------- per-pair mix
  const perPair = tally(touched, (pair) => pair.added.length);
  let spanning = 0;
  let positionVaried = 0;
  for (const pair of touched) {
    const categories = new Set([...pair.original, ...pair.added].map((s) => s.category));
    if (categories.size >= 2) spanning++;
    const relations = new Set([...pair.original, ...pair.added].map((s) => s.relation));
    if (relations.size >= 2) positionVaried++;
  }

  // ------------------------------------------------------------------------------ physics
  //
  // Measured against the original samples as a baseline. A flag rate means nothing on its
  // own — the solver flags conservatively by design — so the question is never "is 20%
  // nested high" but "is it higher than what this pipeline already produces".
  const physicsOf = (rows) => {
    const placed = rows.filter((s) => Array.isArray(s.combined_size));
    const flags = {};
    const contacts = {};
    for (const sample of placed) {
      const contact = sample.physics?.contact ?? sample.intent?.contact ?? 'unknown';
      contacts[contact] = (contacts[contact] ?? 0) + 1;
      for (const f of sample.physics?.flags ?? []) flags[f] = (flags[f] ?? 0) + 1;
      if (sample.physics?.error) flags.error = (flags.error ?? 0) + 1;
    }
    return { placed, flags, contacts };
  };
  const { placed: placedSamples, flags, contacts } = physicsOf(backfilled);
  const baseline = physicsOf(original);

  // Relative size is the sharpest cheap plausibility signal: a placed object many times its
  // anchor is a scale failure the phrase cannot excuse.
  const ratiosOf = (rows) =>
    rows
      .map((s) => Math.max(...s.placed.size) / Math.max(...s.anchor.size))
      .filter(Number.isFinite)
      .sort((a, b) => a - b);
  const ratios = ratiosOf(placedSamples);
  const baseRatios = ratiosOf(baseline.placed);
  const quantile = (list, p) => (list.length ? list[Math.floor((list.length - 1) * p)] : NaN);
  const offScale = (list) => list.filter((r) => r > 3 || r < 1 / 20).length;
  const implausible = offScale(ratios);

  // ------------------------------------------------------------------------- phrase checks
  const failures = [];
  for (const sample of backfilled) {
    const problems = recheck(sample, pairs.get(sample.pair));
    if (problems.length) failures.push({ sample, problems });
  }

  // ------------------------------------------------------------------------------- render
  const lines = [];
  const say = (...text) => lines.push(...text);

  say(
    '# Backfill relations — verification report',
    '',
    `- generated: ${new Date().toISOString()}`,
    `- corpus: \`${SOURCE_DIR}\``,
    `- publish prefix: ${published ? `\`${PUBLISH_DIR}\`` : '_not reachable from here — publish column omitted_'}`,
    `- corpus size: **${samples.length}** samples across **${pairs.size}** object pairs`,
    `- of those, **${backfilled.length}** are backfilled relations on **${touched.length}** pairs` +
      ` (${original.length} original)`,
    ...(broken.length ? ['', `> **${broken.length} folder(s) failed to parse** — listed at the end.`] : []),
    '',
    '## 1 · Did every backfilled sample get all the way through',
    '',
    'Each stage only counts a sample the next stage would accept, so a shortfall here is a',
    'gap the next run fills rather than damage.',
    '',
    table(
      ['Stage', 'Sentinel', 'Count', 'Of written'],
      [
        ['Phrase written', '`metadata.json` with `pair`', stage.written, '100%'],
        ['Meshes fetched', '`mesh` on both roles', stage.fetched, pct(stage.fetched, stage.written)],
        ['Placed and baked', '`combined_size`', stage.placed, pct(stage.placed, stage.written)],
        ['Physics recorded', '`physics`', stage.physics, pct(stage.physics, stage.written)],
        ...(published
          ? [['Published', '`metadata.json` on the volume', stage.published, pct(stage.published, stage.written)]]
          : []),
      ],
    ),
    '',
    '## 2 · Category mix',
    '',
    'The corpus total against the pure weight split (rigid ×2, soft ×1.5, the rest ×1). Drift',
    'is expected where a category ran out of eligible pairs and the dealer spread the',
    'remainder; a large negative drift on `soft` or `bonded` is capacity, not a bug.',
    '',
    table(['Category', 'Added', 'Corpus', 'Target', 'Drift', 'Share'], categoryRows),
    '',
    `Added by phrase style: ${Object.entries(tally(backfilled, (s) => s.complexity))
      .sort()
      .map(([k, v]) => `\`${k}\` ${v}`)
      .join(', ')}`,
    `Added by detail level: ${Object.entries(tally(backfilled, (s) => s.detail))
      .sort()
      .map(([k, v]) => `\`${k}\` ${v}`)
      .join(', ')}`,
    `Added by tier: ${Object.entries(tally(backfilled, (s) => s.tier))
      .sort()
      .map(([k, v]) => `\`${k}\` ${v}`)
      .join(', ')}`,
    '',
    '### Relations used',
    '',
    table(
      ['Relation', 'Added'],
      Object.entries(tally(backfilled, (s) => s.relation))
        .sort((a, b) => b[1] - a[1])
        .map(([relation, n]) => [`\`${relation}\``, n]),
    ),
    '',
    '## 3 · Per-pair variety',
    '',
    'The whole point of the backfill: a pair seen once teaches "these two go like this".',
    '',
    table(
      ['Measure', 'Count', 'Of pairs touched'],
      [
        ['Pairs given new relations', touched.length, '100%'],
        ['…now spanning ≥2 categories', spanning, pct(spanning, touched.length)],
        ['…now carrying ≥2 distinct relations', positionVaried, pct(positionVaried, touched.length)],
      ],
    ),
    '',
    `New relations per pair: ${Object.entries(perPair)
      .sort(([a], [b]) => a - b)
      .map(([n, c]) => `${n}×${c} pair(s)`)
      .join(', ')}`,
    '',
    '## 4 · Physics outcomes',
    '',
    placedSamples.length
      ? [
          table(
            ['Contact', 'Backfilled', 'Share', 'Original corpus'],
            Object.entries(contacts)
              .sort((a, b) => b[1] - a[1])
              .map(([k, v]) => [
                `\`${k}\``,
                v,
                pct(v, placedSamples.length),
                pct(baseline.contacts[k] ?? 0, baseline.placed.length),
              ]),
          ),
          '',
          Object.keys(flags).length
            ? [
                'Against the original samples as a baseline — the solver flags conservatively by',
                'design, so what matters is whether the backfill is worse than what the pipeline',
                'already produced, not the absolute rate.',
                '',
                table(
                  ['Flag', 'Backfilled', 'Share', 'Original corpus', 'Delta'],
                  Object.entries(flags)
                    .sort((a, b) => b[1] - a[1])
                    .map(([k, v]) => {
                      const mine = (v / Math.max(placedSamples.length, 1)) * 100;
                      const theirs = ((baseline.flags[k] ?? 0) / Math.max(baseline.placed.length, 1)) * 100;
                      const delta = mine - theirs;
                      return [
                        `\`${k}\``,
                        v,
                        `${mine.toFixed(1)}%`,
                        `${theirs.toFixed(1)}%`,
                        `${delta >= 0 ? '+' : ''}${delta.toFixed(1)} pp`,
                      ];
                    }),
                ),
              ].join('\n')
            : '_No physics flags raised — every solve was clean._',
          '',
          '### Relative size sanity',
          '',
          'Longest axis of the placed object over the anchor\'s, after the model chose the scale.',
          '',
          table(
            ['Samples', 'p5', 'p25', 'median', 'p75', 'p95', 'implausible (>3× or <1/20×)'],
            [
              [
                'Backfilled',
                ...[0.05, 0.25, 0.5, 0.75, 0.95].map((p) => quantile(ratios, p).toFixed(3)),
                `${implausible} (${pct(implausible, ratios.length)})`,
              ],
              [
                'Original corpus',
                ...[0.05, 0.25, 0.5, 0.75, 0.95].map((p) => quantile(baseRatios, p).toFixed(3)),
                `${offScale(baseRatios)} (${pct(offScale(baseRatios), baseRatios.length)})`,
              ],
            ],
          ),
        ].join('\n')
      : '_Nothing placed yet._',
    '',
    '## 5 · Automated phrase checks',
    '',
    'The writer\'s own rules, re-run against what landed on disk: both objects named, the',
    'anchor written as "the …", the assigned relation present verbatim, no banned verb, no',
    'exact duplicate within the pair.',
    '',
    failures.length
      ? [
          `**${failures.length} of ${backfilled.length} failed.**`,
          '',
          table(
            ['Sample', 'Phrase', 'Problem'],
            failures.slice(0, 100).map((f) => [`\`${f.sample.id}\``, `"${meta.phraseOf(f.sample)}"`, f.problems.join('; ')]),
          ),
        ].join('\n')
      : `**All ${backfilled.length} passed.**`,
    '',
    '## 6 · Every new placement, grouped by pair',
    '',
    'The pair\'s original phrase comes first, then what the backfill added. Read down a block',
    'to judge whether the new placements are plausible for those two objects and genuinely',
    'different from the original.',
    '',
  );

  const shown = LIMIT_EXAMPLES > 0 ? touched.slice(0, LIMIT_EXAMPLES) : touched;
  for (const pair of shown) {
    say(`### ${pair.placed} × ${pair.anchor}`, '');
    for (const sample of pair.original) {
      say(`- _original_ · \`${sample.category}\` · \`${sample.relation}\` — "${meta.phraseOf(sample)}"`);
    }
    for (const sample of pair.added) {
      const size = Array.isArray(sample.combined_size) ? '' : ' · _not placed yet_';
      const physics = sample.physics?.flags?.length ? ` · flags: ${sample.physics.flags.join(', ')}` : '';
      say(
        `- **new** · \`${sample.category}\` · \`${sample.relation}\` · \`${sample.detail}\` · \`${sample.tier}\`` +
          ` — "${meta.phraseOf(sample)}"${size}${physics}`,
      );
    }
    say('');
  }
  if (shown.length < touched.length) {
    say(`_… and ${touched.length - shown.length} more pair(s). Re-run with \`--limit-examples=0\` for all._`, '');
  }

  if (broken.length) {
    say('## 7 · Unreadable folders', '');
    for (const entry of broken.slice(0, 50)) say(`- \`${entry.id}\` — ${entry.broken}`);
    say('');
  }

  return { text: lines.join('\n'), stage, backfilled: backfilled.length, touched: touched.length, failures };
}

const report = await build();
meta.writeAtomic(OUT, report.text);
console.log(
  `\n${report.backfilled} backfilled sample(s) on ${report.touched} pair(s)` +
    ` — ${report.stage.placed} placed, ${report.stage.published} published, ${report.failures.length} phrase check failure(s)`,
);
console.log(`report → ${OUT}`);
if (report.failures.length) process.exitCode = 1;
