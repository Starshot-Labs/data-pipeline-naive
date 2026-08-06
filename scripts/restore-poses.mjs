// Restores baked poses from placement logs — the audit trail earning its keep.
//
//   node scripts/restore-poses.mjs --source=<corpus> --before=2026-07-30T14-00-00-000Z
//
// Every placement writes its resolved transforms into `placement-logs/` before baking, so
// a corpus whose posed copies were overwritten can be wound back: for each sample, the most
// recent log strictly older than `--before` (timestamps in the dashed filename format) is
// replayed through the same bake the pipeline uses. Raw meshes are untouched as always;
// the posed copies and the measured sizes in metadata are rewritten to match the old pose.
//
// The physics report and intent recorded at the original bake are not reconstructable from
// the log's summary text, so those fields are dropped rather than left describing poses
// they no longer belong to.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { bakeDir } from '../pipeline/ops.mjs';
import * as meta from '../pipeline/metadata.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const args = process.argv.slice(2);
const flag = (name) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3);

const SOURCE = path.resolve(ROOT, flag('source') ?? '');
const LOG_DIR = path.resolve(ROOT, flag('logs') ?? 'placement-logs');
const BEFORE = flag('before');
if (!flag('source') || !BEFORE) {
  console.error('usage: node scripts/restore-poses.mjs --source=<corpus> --before=<dashed timestamp>');
  process.exit(2);
}

const logs = fs.readdirSync(LOG_DIR).filter((file) => file.endsWith('.md'));

/** The newest log for `id` from before the cutoff, or null. Filenames sort by time. */
function logFor(id) {
  const mine = logs
    .filter((file) => file.endsWith(`_${id}.md`) && file.slice(0, file.indexOf('_')) < BEFORE)
    .sort();
  return mine.length ? path.join(LOG_DIR, mine.at(-1)) : null;
}

function transformsFrom(file) {
  const text = fs.readFileSync(file, 'utf8');
  const heading = text.indexOf('## Resolved transforms');
  const start = text.indexOf('```json', heading);
  const end = text.indexOf('```', start + 7);
  if (heading < 0 || start < 0 || end < 0) throw new Error('no resolved transforms block');
  return JSON.parse(text.slice(start + 7, end));
}

let restored = 0;
let missing = 0;
for (const sample of meta.list(SOURCE)) {
  const file = logFor(sample.id);
  if (!file) {
    missing++;
    console.error(`  ✗ ${sample.id}: no log before ${BEFORE}`);
    continue;
  }

  const transforms = transformsFrom(file);
  const stem = meta.stems(sample.metadata);
  const { sizes, combined_size } = bakeDir(
    sample.dir,
    sample.dir,
    meta.ROLES.map((role) => ({ role, stem: stem[role], trs: transforms[role] })),
  );

  for (const role of meta.ROLES) sample.metadata[role].size = sizes[role];
  sample.metadata.combined_size = combined_size;
  delete sample.metadata.physics;
  delete sample.metadata.intent;
  delete sample.metadata.scale_estimate;
  meta.write(sample.dir, sample.metadata);

  restored++;
  console.log(`  ↩ ${sample.id}  from ${path.basename(file)}`);
}

console.log(`\n${restored} restored, ${missing} without a usable log`);
