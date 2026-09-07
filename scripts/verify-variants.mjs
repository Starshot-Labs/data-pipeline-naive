// Corpus-wide audit of the four placement forms, run inside a Modal container against the
// mounted volume: how many samples carry them, in both staging and the published dataset,
// and whether placement.txt agrees with metadata.json.
import fs from 'node:fs';
import path from 'node:path';
import { mapLimit } from '/app/pipeline/limit.mjs';
import * as meta from '/app/pipeline/metadata.mjs';
import { track } from '/app/pipeline/progress.mjs';

const roots = { staging: '/scene/datasets/raw/staging', stage1: '/scene/datasets/raw/stage1' };

for (const [label, root] of Object.entries(roots)) {
  const ids = fs.readdirSync(root, { withFileTypes: true }).filter((e) => e.isDirectory()).map((e) => e.name);
  const progress = track(`${label}`, ids.length);
  const stats = { total: ids.length, withVariants: 0, single: 0, noMeta: 0, txtMatches: 0, txtMissing: 0, txtWrong: 0 };

  await mapLimit(ids, 64, async (id) => {
    try {
      const metadata = JSON.parse(await fs.promises.readFile(path.join(root, id, meta.FILE), 'utf8'));
      if (meta.hasVariants(metadata)) stats.withVariants++;
      else stats.single++;
      try {
        const txt = await fs.promises.readFile(path.join(root, id, 'placement.txt'), 'utf8');
        if (txt === meta.placementText(metadata)) stats.txtMatches++;
        else stats.txtWrong++;
      } catch {
        stats.txtMissing++;
      }
    } catch {
      stats.noMeta++;
    } finally {
      progress.tick();
    }
  });
  progress.done();
  console.log(`  ${label}: ${JSON.stringify(stats)}`);
}
