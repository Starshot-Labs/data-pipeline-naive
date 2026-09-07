// The one definition of a sample's metadata.json. Field meanings live in METADATA.md.
//
// The published sample, on the scene volume, is self-contained:
//
//   <anchor slug>_<uuid>/
//     metadata.json
//     placement.txt              the phrase and its three shortened forms, mirroring `placement`
//     <anchor slug>_<uuid>.png   <anchor slug>_<uuid>.glb
//     <placed slug>_<uuid>.png   <placed slug>_<uuid>.glb
//
// Locally a sample folder holds only the small half. The invented-object flow keeps its two
// reference images there because it makes them here and has to send them up to the farm; the
// seeded flow renders beside the volume, so only the two text files stay. Either way the
// meshes never come down — `mesh` names the file as it exists on the volume, and is recorded
// by the stage that produced it rather than discovered on disk.

import fs from 'node:fs';
import path from 'node:path';
import { mapLimit, widthOf } from './limit.mjs';
import { track } from './progress.mjs';

export const FILE = 'metadata.json';
export const ROLES = ['anchor', 'placed'];
export const IMAGE_EXT = /\.(png|jpe?g|webp)$/i;

/**
 * The canonical key order, top level and inside a role block.
 *
 * Stages assign fields as they land and `JSON.stringify` follows insertion order, so without
 * this the same format serialises differently depending on the route a sample took — the
 * fetch records a mesh before its image, a re-place writes `size` again after both. Ordering
 * at the one point everything writes through means every file in a corpus reads the same way,
 * whichever stage last touched it.
 *
 * The order is chronological, which is also how the fields group: what stage 1 invents, what
 * the fetch adds, what the bake measures. A key not listed here is kept and written last, so
 * a corpus from the invented-object flow keeps its `phrasing` rather than losing it to a
 * format that no longer has the field.
 */
const ORDER = [
  'id', 'uuid', 'created_at', 'context', 'pair', 'tier',
  'category', 'complexity', 'relation', 'detail',
  'placement', 'placement_original', 'anchor', 'placed', 'combined_size', 'intent', 'physics',
];
const ROLE_ORDER = ['name', 'description', 'objaverse', 'image', 'mesh', 'textured', 'size'];

// Unlisted keys all rank equal and the sort is stable, so they keep the order they arrived in.
const rank = (order, key) => (order.includes(key) ? order.indexOf(key) : order.length);

const reorder = (object, order) =>
  Object.fromEntries(
    Object.keys(object)
      .sort((a, b) => rank(order, a) - rank(order, b))
      .map((key) => [key, object[key]]),
  );

/** `metadata` with every key in canonical order, role blocks included. */
function ordered(metadata) {
  const sorted = reorder(metadata, ORDER);
  for (const role of ROLES) {
    if (sorted[role]) sorted[role] = reorder(sorted[role], ROLE_ORDER);
  }
  return sorted;
}

/** The exact bytes of a metadata.json — the local copy and the published one are the same
 *  file, so both go through here rather than each stringifying for itself. */
export const serialize = (metadata) => `${JSON.stringify(ordered(metadata), null, 2)}\n`;

/**
 * The placement phrase, in every shape the corpus has ever written it.
 *
 * A sample carries four forms of its phrase — as written, then without the anchor named,
 * without the placed object, and without either — so a trained model sees the same pose
 * described at four levels of grounding. They live in `placement` as an array, with
 * `placement_original` holding the first on its own.
 *
 * These accessors exist because the corpus is permanently mixed: samples written before the
 * variants carry a bare string, and both shapes are valid on disk. Everything that wants
 * *the* phrase — the placement prompt, dedup keys, logs — goes through `phraseOf`, and
 * nothing reads `.placement` directly, or it eventually meets a four-line array where it
 * expected a sentence.
 */
export const variantsOf = (metadata) =>
  Array.isArray(metadata.placement) ? metadata.placement.filter(Boolean) : [metadata.placement].filter(Boolean);

export const phraseOf = (metadata) =>
  (metadata.placement_original ?? variantsOf(metadata)[0] ?? '').trim();

/** Whether the shortened forms have been written yet. */
export const hasVariants = (metadata) => variantsOf(metadata).length > 1;

/** The exact bytes of a placement.txt — the four forms, one per line. */
export const placementText = (metadata) => `${variantsOf(metadata).join('\n')}\n`;

/** Records the three shortened forms against the phrase the sample was born with. */
export function setVariants(metadata, [withoutAnchor, withoutPlaced, withoutBoth]) {
  const original = phraseOf(metadata);
  metadata.placement = [original, withoutAnchor.trim(), withoutPlaced.trim(), withoutBoth.trim()];
  metadata.placement_original = original;
  return metadata;
}

export const slug = (name) => name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '') || 'object';

/**
 * The `<slug>_<uuid>` file stem for each role, which every one of a sample's files is named
 * after. The placed object takes a `_b` suffix when both slugs come out the same, and the
 * anchor's stem is the sample id by construction.
 */
export function stems({ uuid, anchor, placed }) {
  const a = slug(anchor.name);
  const b = slug(placed.name);
  return { anchor: `${a}_${uuid}`, placed: `${b === a ? `${b}_b` : b}_${uuid}` };
}

/**
 * Reference images are normalised to PNG as they come off the model, so a sample's filenames
 * are fully derivable from its stems. `IMAGE_EXT` still exists because reads stay tolerant of
 * a corpus rendered before that was true — treating an old `.jpg` as a missing image would
 * quietly re-render it, and images are the one part of this pipeline billed per attempt.
 */
export const imageName = (stem) => `${stem}.png`;

/**
 * Writes through a temporary name and renames into place.
 *
 * The rename is atomic, so anything reading concurrently — including a Modal volume commit
 * taken while the pipeline is still working — sees either the old file or the whole new one,
 * never half of one. It matters most for metadata.json, where a truncated file fails to parse
 * and takes its sample out of every later stage.
 */
export function writeAtomic(file, contents) {
  const temporary = `${file}.tmp`;
  fs.writeFileSync(temporary, contents);
  fs.renameSync(temporary, file);
}

/**
 * The same, off the event loop.
 *
 * A stage rewriting tens of thousands of samples cannot use the synchronous form: every
 * write and rename is a round trip to a network volume, and blocking on them one at a time
 * serialises the whole pass however wide its concurrency claims to be — twenty thousand
 * samples went from an hour and a half to minutes on this alone.
 */
export async function writeAtomicAsync(file, contents) {
  const temporary = `${file}.tmp`;
  await fs.promises.writeFile(temporary, contents);
  await fs.promises.rename(temporary, file);
}

export const writeAsync = (dir, metadata) => writeAtomicAsync(path.join(dir, FILE), serialize(metadata));

export const exists = (dir) => fs.existsSync(path.join(dir, FILE));

/** Named in the error, because a parse failure otherwise says nothing about which of
 *  hundreds of samples is the broken one, and every stage starts by reading all of them. */
export function read(dir) {
  const file = path.join(dir, FILE);
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (err) {
    throw new Error(`${file}: ${err.message}`);
  }
}
export const write = (dir, metadata) => writeAtomic(path.join(dir, FILE), serialize(metadata));

/** A sample is finished once it has been placed, which is the last thing to land. */
export const isPlaced = (metadata) => Array.isArray(metadata.combined_size);

/** Whether a mesh has been collected onto the volume for both roles. */
export const isMeshed = (metadata) => ROLES.every((role) => metadata[role].mesh);

/** Records what a farm campaign collected. Textures follow the lane each role was sent down,
 *  which is a fact about that campaign — a seeded asset arrives with its own materials and is
 *  recorded textured for both roles by the fetch stage instead. */
export function recordMesh(metadata, role, mesh) {
  metadata[role].mesh = mesh;
  metadata[role].textured = role === 'placed';
  return metadata;
}

/**
 * Rewrites the image fields from what is actually on disk.
 *
 * Images are made here, so for them the local filesystem stays the source of truth and this
 * is idempotent — a crash mid-stage costs nothing, since whatever landed is picked up on the
 * next pass. Meshes are not: they are collected volume-to-volume and never arrive here, so
 * `mesh` and everything downstream of it is written by the stage that produced it.
 */
export function syncImages(dir, metadata) {
  const entries = fs.readdirSync(dir);
  const stem = stems(metadata);

  for (const role of ROLES) {
    const image = entries.find((file) => file.startsWith(`${stem[role]}.`) && IMAGE_EXT.test(file));
    if (image) metadata[role].image = image;
    else delete metadata[role].image;
  }
  return metadata;
}

/**
 * Every sample folder under `root`, newest field values read fresh off disk.
 *
 * `only` narrows the read to named ids instead of walking the corpus. A sharded run hands
 * each worker a slice, and without this every one of them would read every sample's
 * metadata to find its own — twelve shards over 20k samples is a quarter of a million
 * volume reads to do the work of twenty thousand.
 */
export function list(root, only = null) {
  if (!fs.existsSync(root)) return [];
  const names = only
    ? [...only].filter((name) => exists(path.join(root, name)))
    : fs
        .readdirSync(root, { withFileTypes: true })
        .filter((entry) => entry.isDirectory() && exists(path.join(root, entry.name)))
        .map((entry) => entry.name);

  return names.map((name) => {
    const dir = path.join(root, name);
    return { id: name, dir, metadata: read(dir) };
  });
}

/**
 * `list`, but reading concurrently and saying how it is going.
 *
 * `list` is synchronous, which is fine for hundreds and ruinous for tens of thousands: on a
 * network volume none of them are cached, so it becomes that many serial round trips —
 * twenty thousand is over a quarter of an hour with nothing printed, which reads as a hung
 * process rather than a slow one. Overlapping the waits is the whole difference, and the
 * tracker is there so the wait is legible while it happens.
 *
 * Note that `fs.promises` runs on libuv's threadpool, four threads by default, so a
 * concurrency above `UV_THREADPOOL_SIZE` buys nothing. The Modal image raises it to match.
 */
export async function listAsync(root, { only = null, concurrency = null, label = 'reading corpus' } = {}) {
  if (!fs.existsSync(root)) return [];
  const names = only
    ? [...only]
    : fs
        .readdirSync(root, { withFileTypes: true })
        .filter((entry) => entry.isDirectory())
        .map((entry) => entry.name);

  const width = concurrency ?? widthOf('CORPUS_CONCURRENCY', 64);
  const progress = track(label, names.length);
  const broken = [];
  const samples = await mapLimit(names, width, async (name) => {
    const dir = path.join(root, name);
    let raw;
    try {
      raw = await fs.promises.readFile(path.join(dir, FILE), 'utf8');
    } catch (err) {
      // No metadata.json means the folder is not a sample, which is exactly what `list`'s
      // own filter decides.
      if (err.code !== 'ENOENT') broken.push(`${name}: ${err.message}`);
      return null;
    } finally {
      progress.tick();
    }

    try {
      return { id: name, dir, metadata: JSON.parse(raw) };
    } catch (err) {
      // A corrupt sample is not a sample, and it is emphatically not a reason to abandon
      // the eighty thousand beside it — a container killed mid-write can leave a
      // zero-filled metadata.json, and letting that throw takes down a whole stage.
      broken.push(`${name}: ${err.message}`);
      return null;
    }
  });
  progress.done();
  if (broken.length) {
    console.error(`  ✗ ${broken.length} unreadable sample(s), skipped:`);
    for (const line of broken.slice(0, 10)) console.error(`      ${line}`);
    if (broken.length > 10) console.error(`      … and ${broken.length - 10} more`);
  }
  return samples.filter(Boolean);
}

/** Persists `metadata` only when the change is real, so re-syncing costs no writes. */
export function writeIfChanged(dir, metadata, before) {
  if (JSON.stringify(metadata) === before) return false;
  write(dir, metadata);
  return true;
}

/** Syncs every sample's images against disk and persists the ones that changed. */
export function syncAll(root) {
  const samples = list(root);
  for (const sample of samples) {
    const before = JSON.stringify(sample.metadata);
    syncImages(sample.dir, sample.metadata);
    writeIfChanged(sample.dir, sample.metadata, before);
  }
  return samples;
}
