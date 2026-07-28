// Steps 1-3: invent samples, render their reference images, build the meshes.
// Every stage skips work that is already on disk, so re-running fills the gaps.
//
//   node pipeline/build.mjs --pairs=50   invent 50 pairs, then advance everything
//   node pipeline/build.mjs              advance existing samples only
//
// Each sample lands as one self-contained folder described by METADATA.md. Steps 4-6 read
// those folders with `node pipeline/run.mjs`, or run the lot with `node pipeline/all.mjs 50`.

import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { proposeSamples, planRequests, CONTEXTS } from './generate.mjs';
import { ANCHOR_VIEW, randomView, imagePrompt, renderImage } from './images.mjs';
import { generateMeshes } from './trellis.mjs';
import { mapLimit, retry, widthOf } from './limit.mjs';
import * as meta from './metadata.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ENV_FILE = path.join(ROOT, '.env');
if (fs.existsSync(ENV_FILE)) process.loadEnvFile(ENV_FILE);

const args = process.argv.slice(2);
const flag = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

export const GENERATED_DIR = path.resolve(ROOT, flag('out', process.env.GENERATED_DIR ?? 'generated'));
const BATCH = Number(process.env.SPEC_BATCH ?? 20);
const SPEC_WIDTH = widthOf('SPEC_CONCURRENCY');
const IMAGE_WIDTH = widthOf('IMAGE_CONCURRENCY');

const findImage = (dir, stem) =>
  fs.readdirSync(dir).find((file) => file.startsWith(`${stem}.`) && meta.IMAGE_EXT.test(file)) ?? null;

const keyOf = (anchor, placed, placement) =>
  `${meta.slug(anchor)}|${meta.slug(placed)}|${placement.trim().toLowerCase()}`;

/** One sample folder, holding nothing yet but what stage 1 knows. */
function writeSample(proposal, context, phrasing) {
  const uuid = randomUUID();
  const id = `${meta.slug(proposal.anchor_name)}_${uuid}`;
  const dir = path.join(GENERATED_DIR, id);
  fs.mkdirSync(dir, { recursive: true });

  const metadata = {
    id,
    uuid,
    created_at: new Date().toISOString(),
    context,
    phrasing,
    placement: proposal.placement.trim(),
    anchor: { name: proposal.anchor_name, description: proposal.anchor_description },
    placed: { name: proposal.object_name, description: proposal.object_description },
  };

  meta.write(dir, metadata);
  meta.writeAtomic(path.join(dir, 'placement.txt'), metadata.placement);
  return metadata;
}

async function inventSamples(target) {
  const existing = meta.list(GENERATED_DIR).map((sample) => sample.metadata);
  const seen = new Set(existing.map((m) => keyOf(m.anchor.name, m.placed.name, m.placement)));
  const anchors = [...new Set(existing.map((m) => m.anchor.name))];

  let created = 0;
  // Duplicates get dropped, so a round can come up short — re-plan the shortfall.
  for (let round = 0; created < target && round < 6; round++) {
    const avoid = anchors.slice(-60);
    await mapLimit(planRequests(target - created, BATCH), SPEC_WIDTH, async ({ context, count, phrasing }) => {
      let samples;
      try {
        samples = await retry(() => proposeSamples({ count, context, phrasing, avoid }));
      } catch (err) {
        console.error(`  ✗ ${context}: ${err.message}`);
        return;
      }

      // Written as each batch lands rather than after the whole round, so a rate limit
      // later on cannot throw away specs that already came back. Nothing awaits inside
      // this loop, so the dedupe check and the write cannot interleave with another worker.
      for (const proposal of samples) {
        if (created >= target) return;
        const key = keyOf(proposal.anchor_name, proposal.object_name, proposal.placement);
        if (seen.has(key)) continue;
        seen.add(key);
        const metadata = writeSample(proposal, context, phrasing);
        anchors.push(metadata.anchor.name);
        created++;
        console.log(`  + [${context}/${phrasing}] ${metadata.placed.name} → ${metadata.placement}`);
      }
    });
  }
  return created;
}

/**
 * The anchor is always shot straight on and the placed object from an arbitrary direction.
 * The angle is drawn here rather than stored, so an image that failed and gets retried
 * simply draws a fresh one.
 */
const viewFor = (role) => (role === 'anchor' ? ANCHOR_VIEW : randomView());

async function renderMissingImages() {
  const jobs = meta.list(GENERATED_DIR).flatMap((sample) => {
    const stem = meta.stems(sample.metadata);
    return meta.ROLES.filter((role) => !findImage(sample.dir, stem[role])).map((role) => ({
      id: sample.id,
      dir: sample.dir,
      role,
      stem: stem[role],
      description: sample.metadata[role].description,
    }));
  });
  if (!jobs.length) return 0;

  console.log(`  rendering ${jobs.length} image(s)…`);
  const done = await mapLimit(jobs, IMAGE_WIDTH, async (job) => {
    try {
      // The view is drawn inside the retry, so a re-roll also gets a fresh angle rather
      // than asking again for the one the model just declined to produce.
      const png = await retry(() => renderImage({ prompt: imagePrompt(job.description, viewFor(job.role)) }));
      meta.writeAtomic(path.join(job.dir, meta.imageName(job.stem)), png);
      return true;
    } catch (err) {
      console.error(`  ✗ ${job.id} [${job.role}]: ${err.message}`);
      return false;
    }
  });
  return done.filter(Boolean).length;
}

/**
 * One job per mesh that has an image but no mesh recorded yet. The mesh itself lives on the
 * volume, so what says it exists is `metadata[role].mesh` rather than anything on disk here.
 */
function meshJobs() {
  return meta.list(GENERATED_DIR).flatMap((sample) => {
    const stem = meta.stems(sample.metadata);
    return meta.ROLES.flatMap((role) => {
      const image = findImage(sample.dir, stem[role]);
      return sample.metadata[role].mesh || !image
        ? []
        : [{ id: sample.id, role, stem: stem[role], image: path.join(sample.dir, image) }];
    });
  });
}

/** Writes what the collect step landed back into each sample's metadata. */
function recordMeshes(collected) {
  const bySample = new Map();
  for (const item of collected) {
    if (!bySample.has(item.id)) bySample.set(item.id, []);
    bySample.get(item.id).push(item);
  }

  for (const [id, items] of bySample) {
    const dir = path.join(GENERATED_DIR, id);
    if (!meta.exists(dir)) continue;
    const metadata = meta.read(dir);
    const before = JSON.stringify(metadata);
    for (const item of items) meta.recordMesh(metadata, item.role, item.mesh);
    meta.writeIfChanged(dir, metadata, before);
  }
}

/** Refreshes every sample's metadata from disk, then reports how many pass `ready`. */
function settle(ready, label) {
  const samples = meta.syncAll(GENERATED_DIR);
  const done = samples.filter((sample) => ready(sample.metadata));
  console.log(`\n${samples.length} sample(s) in ${GENERATED_DIR}, ${done.length} ${label}.`);
  return done.map((sample) => sample.id);
}

/** Stages 1-2. Returns the ids of samples that now have every image they need. */
export async function buildImages({ pairs = 0 } = {}) {
  fs.mkdirSync(GENERATED_DIR, { recursive: true });

  if (pairs > 0) {
    console.log(`\n1 · inventing ${pairs} pair(s) across ${CONTEXTS.length} contexts`);
    console.log(`  created ${await inventSamples(pairs)}`);
  }

  console.log('\n2 · reference images');
  console.log(`  rendered ${await renderMissingImages()}`);

  return settle((m) => meta.ROLES.every((role) => m[role].image), 'with both images');
}

/**
 * A pair is only worth anything with both of its meshes, so one the farm has given up on
 * takes the whole sample with it rather than sitting there being re-uploaded every run.
 */
function discardSamples(jobs) {
  const ids = [...new Set(jobs.map((job) => job.id))];
  for (const id of ids) {
    fs.rmSync(path.join(GENERATED_DIR, id), { recursive: true, force: true });
    console.log(`  – discarded ${id}: the farm dropped one of its meshes`);
  }
  return ids.length;
}

/** Stage 3. Returns the ids of samples that now have both meshes on the volume. */
export async function buildMeshes() {
  fs.mkdirSync(GENERATED_DIR, { recursive: true });

  console.log('\n3 · meshes');
  const jobs = meshJobs();
  if (!jobs.length) {
    console.log('  nothing to build');
  } else {
    const { collected, abandoned } = await generateMeshes(jobs);
    recordMeshes(collected);
    discardSamples(abandoned);
  }

  return settle(meta.isMeshed, 'with meshes');
}

/** Stages 1-3 in one go. */
export async function buildSamples({ pairs = 0 } = {}) {
  await buildImages({ pairs });
  return buildMeshes();
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const ready = await buildSamples({ pairs: Number(flag('pairs', 0)) });
  if (ready.length) console.log('Place them with:  node pipeline/run.mjs');
}
