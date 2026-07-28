// The one definition of a sample's metadata.json. Field meanings live in METADATA.md.
//
// The published sample, on the scene volume, is self-contained:
//
//   <anchor slug>_<uuid>/
//     metadata.json
//     placement.txt              the phrase on its own, mirroring `placement`
//     <anchor slug>_<uuid>.png   <anchor slug>_<uuid>.glb
//     <placed slug>_<uuid>.png   <placed slug>_<uuid>.glb
//
// Locally a sample folder holds only the small half — metadata.json, placement.txt and the
// two reference images, which are made here and have to go up to the farm. The meshes never
// come down; `mesh` names the file as it exists on the volume, and is recorded by stage 3
// rather than discovered on disk.

import fs from 'node:fs';
import path from 'node:path';

export const FILE = 'metadata.json';
export const ROLES = ['anchor', 'placed'];
export const IMAGE_EXT = /\.(png|jpe?g|webp)$/i;

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
export const write = (dir, metadata) => writeAtomic(path.join(dir, FILE), `${JSON.stringify(metadata, null, 2)}\n`);

/** A sample is finished once it has been placed, which is the last thing to land. */
export const isPlaced = (metadata) => Array.isArray(metadata.combined_size);

/** Whether a mesh has been collected onto the volume for both roles. */
export const isMeshed = (metadata) => ROLES.every((role) => metadata[role].mesh);

/** Records what stage 3 collected. Textures follow the lane each role was sent down. */
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

/** Every sample folder under `root`, newest field values read fresh off disk. */
export function list(root) {
  if (!fs.existsSync(root)) return [];
  return fs
    .readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && exists(path.join(root, entry.name)))
    .map((entry) => {
      const dir = path.join(root, entry.name);
      return { id: entry.name, dir, metadata: read(dir) };
    });
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
