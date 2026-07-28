// Steps 4-6: voxelize both meshes, ask an LLM where the placed object goes, then bake that
// answer into the two GLBs and record the sizes it produces.
//
//   node pipeline/run.mjs                    every unplaced sample in generated/
//   node pipeline/run.mjs <sampleId> ...     specific samples
//   node pipeline/run.mjs --source=other     read a different folder of samples
//   node pipeline/run.mjs --dry              print the LLM prompt, call no model
//
// The meshes are not here — they live on the scene volume, and `dc-scene-ops` voxelizes and
// bakes them in place. What crosses this machine is the occupancy grids on the way to the
// model and the resolved transform on the way back, which is a few hundred KB against the
// several MB the GLBs would have cost in each direction.
//
// Baking reads the raw mesh and writes a posed copy elsewhere, so it is idempotent: a sample
// can be re-placed as many times as you like without re-meshing it. `combined_size` in the
// metadata is what marks one done.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Vector3, Quaternion, MathUtils } from 'three';
import { requestPlacement, buildPrompt } from './place.mjs';
import { mapLimit, retry, widthOf } from './limit.mjs';
import * as scene from './scene.mjs';
import * as meta from './metadata.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ENV_FILE = path.join(ROOT, '.env');
if (fs.existsSync(ENV_FILE)) process.loadEnvFile(ENV_FILE);

const args = process.argv.slice(2);
const flag = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

const SOURCE_DIR = path.resolve(ROOT, flag('source', process.env.GENERATED_DIR ?? 'generated'));
const RESOLUTION = {
  anchor: Number(process.env.VOXEL_RES_ANCHOR ?? 64),
  placed: Number(process.env.VOXEL_RES_PLACED ?? 32),
};
const PLACE_WIDTH = widthOf('PLACE_CONCURRENCY');

const round = (v) => Number(v.toFixed(6));
const rounded = (arr) => arr.map(round);

function readSample(sourceDir, id) {
  const dir = path.join(sourceDir, id);
  if (!meta.exists(dir)) throw new Error(`no ${meta.FILE}`);

  const metadata = meta.read(dir);
  const stem = meta.stems(metadata);
  for (const role of meta.ROLES) {
    if (!metadata[role].mesh) throw new Error(`no ${role} mesh yet`);
  }
  return { dir, metadata, stem };
}

/**
 * The anchor is baked with its bounding-box centre at the origin. The placed object is
 * scaled so one of its voxels spans `scale` anchor voxels, yawed about its own centre, then
 * moved so that centre lands on `center` read in the anchor's grid.
 */
function resolveTransforms(anchor, placed, answer) {
  const scale = (answer.scale * anchor.grid.voxelSize) / placed.grid.voxelSize;
  const rotation = new Quaternion().setFromAxisAngle(new Vector3(0, 1, 0), MathUtils.degToRad(answer.yaw_degrees));
  const target = new Vector3()
    .fromArray(answer.center)
    .multiplyScalar(anchor.grid.voxelSize)
    .add(new Vector3().fromArray(anchor.grid.origin));
  const centreOffset = new Vector3().fromArray(placed.grid.center).multiplyScalar(scale).applyQuaternion(rotation);

  return {
    anchor: {
      position: rounded(new Vector3().fromArray(anchor.grid.center).negate().toArray()),
      rotation: [0, 0, 0, 1],
      scale: [1, 1, 1],
    },
    placed: {
      position: rounded(target.sub(centreOffset).toArray()),
      rotation: rounded(rotation.toArray()),
      scale: rounded([scale, scale, scale]),
    },
  };
}

/** Returns its own log lines so that each sample prints as one block under concurrency. */
async function runSample(sourceDir, id, { dry }) {
  const sample = readSample(sourceDir, id);
  if (meta.isPlaced(sample.metadata)) return [`▸ ${id}`, '  already placed'];

  const grids = await scene.voxelize(
    id,
    meta.ROLES.map((role) => ({ role, stem: sample.stem[role], resolution: RESOLUTION[role] })),
  );
  const objects = Object.fromEntries(
    meta.ROLES.map((role) => [role, { name: sample.metadata[role].name, grid: grids[role], slices: grids[role].slices }]),
  );

  const { anchor, placed } = objects;
  const phrase = sample.metadata.placement;
  console.log(`⋯ ${id}  anchor ${anchor.grid.dims.join('x')}  placed ${placed.grid.dims.join('x')}`);

  if (dry) return [`▸ ${id}`, buildPrompt({ phrase, a: anchor, b: placed })];

  // `requestPlacement` rejects an answer that fails validation, so a retry re-rolls a reply
  // that came back unusable rather than losing the sample until the next pipeline pass.
  const answer = await retry(() => requestPlacement({ phrase, a: anchor, b: placed }));
  const transforms = resolveTransforms(anchor, placed, answer);

  const { sizes, combined_size } = await scene.bake(
    id,
    meta.ROLES.map((role) => ({ role, stem: sample.stem[role], trs: transforms[role] })),
  );
  for (const role of meta.ROLES) sample.metadata[role].size = sizes[role];
  sample.metadata.combined_size = combined_size;
  meta.write(sample.dir, sample.metadata);

  return [
    `▸ ${id}  "${phrase}"`,
    `  scale=${round(answer.scale)}  yaw=${answer.yaw_degrees}  center=[${rounded(answer.center)}]`,
    `  ${answer.reasoning.replace(/\s+/g, ' ').slice(0, 220)}`,
  ];
}

export async function placeSamples({ sourceDir, ids, dry = false, concurrency = PLACE_WIDTH }) {
  // Each sample writes its own metadata before resolving, so a run cut short by rate limits
  // keeps everything that already landed and the next pass skips it.
  const results = await mapLimit(ids, dry ? 1 : concurrency, async (id) => {
    try {
      console.log(`\n${(await runSample(sourceDir, id, { dry })).join('\n')}`);
      return true;
    } catch (err) {
      console.error(`\n▸ ${id}\n  ✗ ${err.message}`);
      return false;
    }
  });

  const placed = results.filter(Boolean).length;
  const done = `${placed}/${ids.length}`;
  console.log(dry ? `\n${done} prompts built` : `\n${done} placed`);
  return ids.length - placed;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const requested = args.filter((arg) => !arg.startsWith('--'));
  // Without both meshes there is nothing to place, and attempting one only reports that.
  // A named id is still tried, so asking about a specific sample still tells you why.
  const ids = requested.length
    ? requested
    : meta.list(SOURCE_DIR).filter((sample) => meta.isMeshed(sample.metadata)).map((sample) => sample.id);
  const failed = await placeSamples({ sourceDir: SOURCE_DIR, ids, dry: args.includes('--dry') });
  if (failed) process.exitCode = 1;
}
