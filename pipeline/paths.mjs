// Where everything on disk lives, and the one place `.env` is read.
//
// Every entry point imports this, so the file is loaded before any stage reads a variable out
// of it — an import is fully evaluated before the importing module's body runs.
//
// Bulk content sits under `data/`, one directory per kind, so the repo root stays code and
// config. Each directory keeps its own environment override; `DATA_DIR` moves the whole lot at
// once, which is what you want when the samples are on another disk.
//
// On Modal this module resolves `ROOT` to `/app`, where there is no `.env` and no `data/`.
// Nothing breaks: `pipeline.py` passes an absolute `GENERATED_DIR`, and an absolute override
// wins over the default outright.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const ENV_FILE = path.join(ROOT, '.env');
if (fs.existsSync(ENV_FILE)) process.loadEnvFile(ENV_FILE);

/** A path from a flag or an environment variable, read from the repo root when relative. */
export const fromRoot = (target) => path.resolve(ROOT, target);

export const DATA_DIR = fromRoot(process.env.DATA_DIR ?? 'data');

const inData = (name, override) => (override ? fromRoot(override) : path.join(DATA_DIR, name));

/** Source GLBs the manual tool loads. */
export const MODELS_DIR = inData('models', process.env.MODELS_DIR);
/** Hand-posed samples the browser exporter writes. */
export const DATASET_DIR = inData('dataset', process.env.DATASET_DIR);
/** The local half of every pipeline sample. */
export const GENERATED_DIR = inData('generated', process.env.GENERATED_DIR);
/** The anchor/photo/phrase inputs both experiments read. */
export const PLACEMENT_SET_DIR = inData('placement-set', process.env.PLACEMENT_SET_DIR);
export const PLACEMENT_RESULTS_DIR = inData('placement-results', process.env.PLACEMENT_RESULTS_DIR);
export const EDIT_RESULTS_DIR = inData('edit-results', process.env.EDIT_RESULTS_DIR);
export const SEGMENT_RESULTS_DIR = inData('segment-results', process.env.SEGMENT_RESULTS_DIR);
/** P3-SAM's automatic instance segmentations, separate from PartField hierarchies. */
export const P3SAM_RESULTS_DIR = inData('p3sam-results', process.env.P3SAM_RESULTS_DIR);
/** The 3D-FRONT rooms the scene-editing benchmark reads, one assembled room per GLB. */
export const FRONT3D_DIR = inData('3d-front', process.env.FRONT3D_DIR);
/** One folder per scene + prompt run, holding a folder of results per model. */
export const SCENE_EDITS_DIR = inData('scene-edits', process.env.SCENE_EDITS_DIR);

/** Where `npm run build` puts the client, and what `npm start` serves. */
export const WEB_DIST_DIR = fromRoot('web/dist');
