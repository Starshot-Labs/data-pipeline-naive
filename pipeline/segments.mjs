// What a segmentation result folder is, shared by the CLI that fills one and the server that
// serves one, so a run started from a terminal and a run started by dropping a file into the
// browser leave the same thing behind:
//
//   segment-results/<id>/
//     parts_NN.glb    one named, coloured node per part, in the source mesh's own frame
//     labels_NN.bin   little-endian int32 per face, the part face i belongs to
//     parts_NN.png    an isometric look at the cut
//     segment.json    the job behind them all, and every cut so far taken of it
//
// NN is the number of parts, because a job is a hierarchy rather than one segmentation and a
// folder accumulates the levels of it anyone has asked for.

import fs from 'node:fs';
import path from 'node:path';
import { parseGLB, sceneTriangles } from './glb.mjs';
import { renderView } from './render.mjs';
import { writeAtomic } from './metadata.mjs';

export const RESULT = 'segment.json';

/** The two-digit level a cut's files are named for. */
export const level = (parts) => String(parts).padStart(2, '0');

/**
 * An isometric shot of the cut, each part in the colour the service gave it.
 *
 * `parts.glb` holds one node per part in part order, so the merged triangle stream splits on
 * the face counts the service reported — checked against the total, since a mismatch would
 * silently colour the picture by the wrong boundaries.
 */
export async function preview({ glb, summary }) {
  const triangles = sceneTriangles(parseGLB(glb));
  if (triangles.length / 9 !== summary.faces) {
    throw new Error(`parts.glb has ${triangles.length / 9} triangles, expected ${summary.faces}`);
  }
  let at = 0;
  const layers = summary.part_faces.map((faces, part) => {
    const layer = { triangles: triangles.subarray(at * 9, (at + faces) * 9), base: summary.part_colors[part] };
    at += faces;
    return layer;
  });
  return renderView(layers);
}

/** Writes one cut into `dir`, and returns the entry describing it. */
export async function writeCut(dir, { glb, labels, summary }) {
  fs.mkdirSync(dir, { recursive: true });
  const at = level(summary.num_parts);
  writeAtomic(path.join(dir, `parts_${at}.glb`), glb);
  writeAtomic(path.join(dir, `labels_${at}.bin`), Buffer.from(labels.buffer));
  writeAtomic(path.join(dir, `parts_${at}.png`), await preview({ glb, summary }));
  return { parts: summary.num_parts, part_faces: summary.part_faces, part_colors: summary.part_colors };
}

/**
 * Records the job and folds `cuts` into whatever a past run left, keyed by their level. A cut
 * is only ever re-taken to be replaced, so the newest wins and the order stays by part count.
 */
export function writeRecord(dir, { id, source, job, summary, cuts }) {
  const previous = readRecord(dir) ?? {};
  const merged = new Map((previous.cuts ?? []).map((entry) => [entry.parts, entry]));
  for (const entry of cuts) merged.set(entry.parts, entry);

  const record = {
    id,
    source,
    job,
    faces: summary.faces,
    vertices: summary.vertices,
    cuts: [...merged.values()].sort((a, b) => a.parts - b.parts),
    levels: summary.levels,
    params: summary.params,
    sampling: summary.sampling,
  };
  writeAtomic(path.join(dir, RESULT), `${JSON.stringify(record, null, 2)}\n`);
  return record;
}

export function readRecord(dir) {
  const file = path.join(dir, RESULT);
  if (!fs.existsSync(file)) return null;
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (err) {
    throw new Error(`${file}: ${err.message}`);
  }
}

/**
 * A folder's record plus the levels actually on disk. The two differ on purpose: `levels` is
 * every cut the service could still answer for the job, `on_disk` only the ones fetched, and
 * the gap between them is what a viewer offers to fill.
 */
export function readResult(dir) {
  const record = readRecord(dir);
  if (!record) return null;
  const on_disk = fs
    .readdirSync(dir)
    .map((file) => /^parts_(\d+)\.glb$/.exec(file))
    .filter(Boolean)
    .map((match) => Number(match[1]))
    .sort((a, b) => a - b);
  return { ...record, on_disk };
}
