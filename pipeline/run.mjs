// Steps 4-6: voxelize both meshes, ask an LLM where the placed object goes, refine that
// answer against the actual mesh surfaces (pipeline/physics.mjs), then bake it into the
// two GLBs and record the sizes it produces.
//
//   node pipeline/run.mjs                    every unplaced sample in generated/
//   node pipeline/run.mjs <sampleId> ...     specific samples
//   node pipeline/run.mjs --source=other     read a different folder of samples
//   node pipeline/run.mjs --dry              print the LLM prompt, call no model
//   node pipeline/run.mjs --force            re-place samples that are already placed
//   node pipeline/run.mjs --physics-only     re-run just the physics on already-placed
//                                            samples, continuing from their baked poses
//
// The meshes are not here — they live on the scene volume, and `dc-scene-ops` voxelizes and
// bakes them in place. What crosses this machine is each object's solid-block decomposition
// on the way to the model and the resolved transform on the way back, which is a few
// hundred KB against the several MB the GLBs would have cost in each direction.
//
// Baking reads the raw mesh and writes a posed copy elsewhere, so it is idempotent: a sample
// can be re-placed as many times as you like without re-meshing it. `combined_size` in the
// metadata is what marks one done.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Vector3, Quaternion, Euler, MathUtils } from 'three';
import { requestPlacement, buildPrompt, SYSTEM, CONTACTS } from './place.mjs';
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
  anchor: Number(process.env.VOXEL_RES_ANCHOR ?? 16),
  placed: Number(process.env.VOXEL_RES_PLACED ?? 16),
};
const PLACE_WIDTH = widthOf('PLACE_CONCURRENCY');
const LOG_DIR = path.resolve(ROOT, process.env.PLACEMENT_LOG_DIR ?? 'placement-logs');
const PHYSICS = (process.env.PLACEMENT_PHYSICS ?? 'on').toLowerCase() !== 'off';
const PHYSICS_OPTIONS = {
  sdfRes: Number(process.env.PHYSICS_SDF_RES ?? 128),
  samples: Number(process.env.PHYSICS_SAMPLES ?? 2000),
  drapeRes: Number(process.env.DRAPE_RES ?? 24),
};
// Overrides the contact the physics acts on — the model's classification (or the stored
// intent, for a physics-only pass) is replaced wholesale. How the viewer's contact
// picker forces, say, a drape onto an object the model called a rest.
const CONTACT_OVERRIDE = CONTACTS.includes(process.env.PLACEMENT_CONTACT) ? process.env.PLACEMENT_CONTACT : null;

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
 * The anchor is baked with its bounding-box centre at the origin, so A's bbox-centred frame
 * — the one the answer speaks — is the world frame. The placed object is scaled by the
 * answer's factor, yawed about its own bbox centre, then moved so that centre lands on
 * `center`.
 */
function resolveTransforms(anchor, placed, answer) {
  // 'YXZ' applies yaw about +Y first, then pitch about the rotated +X, then roll about the
  // resulting +Z — the intrinsic order the placement prompt promises.
  const rotation = new Quaternion().setFromEuler(new Euler(
    MathUtils.degToRad(answer.pitch_degrees),
    MathUtils.degToRad(answer.yaw_degrees),
    MathUtils.degToRad(answer.roll_degrees),
    'YXZ',
  ));
  const target = new Vector3().fromArray(answer.center);
  const centreOffset = new Vector3()
    .fromArray(placed.grid.center)
    .multiplyScalar(answer.scale)
    .applyQuaternion(rotation);

  return {
    anchor: {
      position: rounded(new Vector3().fromArray(anchor.grid.center).negate().toArray()),
      rotation: [0, 0, 0, 1],
      scale: [1, 1, 1],
    },
    placed: {
      position: rounded(target.sub(centreOffset).toArray()),
      rotation: rounded(rotation.toArray()),
      scale: rounded([answer.scale, answer.scale, answer.scale]),
    },
  };
}

/**
 * The pre-physics placed transform, as one structured stdout line the viewer's server
 * relays: the page overlays a translucent ghost of it, so a refinement can be compared
 * against the pose it started from. Behind an env flag so batch runs do not carry a
 * machine line per sample, and only when the physics actually moved something — an
 * unchanged pose has nothing to compare against.
 */
function emitGhost(before, after) {
  if (process.env.PLACEMENT_GHOST !== '1') return;
  const changed =
    before.position.some((v, c) => Math.abs(v - after.position[c]) > 1e-6) ||
    before.rotation.some((v, c) => Math.abs(v - after.rotation[c]) > 1e-6);
  if (changed) console.log(`GHOST ${JSON.stringify(before)}`);
}

/** The one-line human summary of a physics report, shared by the log and the console. */
function physicsSummary(physics) {
  if (!physics) return 'disabled';
  if (physics.error) return `✗ ${physics.error} — baked the transforms unrefined`;
  return (
    `${physics.contact} · moved ${round(physics.moved ?? 0)} · rotated ${physics.rotated_degrees ?? 0}° · ` +
    `gap ${round(physics.gap_before ?? 0)} → ${round(physics.gap_after ?? 0)}` +
    (physics.flags?.length ? ` · flags: ${physics.flags.join(', ')}` : '')
  );
}

/**
 * One Markdown file per placement, written the moment the model answers: the exact system
 * prompt and user prompt it saw in fenced blocks, the full answer it gave, the transforms
 * that answer resolved to, and what the physics pass did to them. Timestamped rather than
 * keyed by sample, so re-placements accumulate instead of overwriting each other. A
 * physics-only pass logs too — `answer` is null then, and the model sections drop out.
 */
function logPlacement(id, phrase, answer, transforms, physics) {
  fs.mkdirSync(LOG_DIR, { recursive: true });
  const stamp = new Date().toISOString();
  const usage = answer?.usage;
  const fence = '```';

  const lines = [
    `# ${id}`,
    '',
    `- at: ${stamp}`,
    ...(answer ? [`- model: ${answer.model}`] : ['- physics-only pass — no model call']),
    ...(usage
      ? [
          `- tokens: ${usage.prompt_tokens} prompt + ${usage.completion_tokens} completion` +
            (usage.completion_tokens_details?.reasoning_tokens
              ? ` (${usage.completion_tokens_details.reasoning_tokens} reasoning)`
              : ''),
          ...(typeof usage.cost === 'number' ? [`- cost: $${usage.cost}`] : []),
        ]
      : []),
    '',
    '## Placement',
    '',
    `"${phrase}"`,
    '',
    ...(answer
      ? [
          '## System prompt',
          '',
          fence,
          SYSTEM,
          fence,
          '',
          '## User prompt',
          '',
          fence,
          answer.prompt,
          fence,
          '',
          '## Answer',
          '',
          `scale ${answer.scale} · yaw ${answer.yaw_degrees}° · pitch ${answer.pitch_degrees}° · roll ${answer.roll_degrees}° · center [${answer.center.join(', ')}]`,
          `contact ${answer.contact ?? '(none given)'} · direction [${(answer.contact_direction ?? []).join(', ')}] · embed ${answer.embed_fraction ?? 0}`,
          '',
          answer.reasoning,
          '',
        ]
      : []),
    '## Physics',
    '',
    physicsSummary(physics),
    '',
    '## Resolved transforms (baked)',
    '',
    `${fence}json`,
    JSON.stringify(transforms, null, 2),
    fence,
    '',
  ];
  meta.writeAtomic(path.join(LOG_DIR, `${stamp.replace(/[:.]/g, '-')}_${id}.md`), lines.join('\n'));
}

/**
 * Records a bake's outcome — sizes, the physics report, and the intent the physics acted
 * on (so a later physics-only pass knows what the placement meant without another model
 * call). Shared by every kind of pass, rigid and soft alike.
 */
function record(sample, { sizes, combined_size }, intent, physics) {
  for (const role of meta.ROLES) sample.metadata[role].size = sizes[role];
  sample.metadata.combined_size = combined_size;
  if (intent) sample.metadata.intent = intent;
  if (physics) {
    sample.metadata.physics = {
      contact: physics.contact,
      ...(physics.error
        ? { error: physics.error }
        : {
            moved: round(physics.moved ?? 0),
            rotated_degrees: physics.rotated_degrees ?? 0,
            gap_before: round(physics.gap_before ?? 0),
            gap_after: round(physics.gap_after ?? 0),
          }),
      flags: physics.flags ?? [],
    };
  } else {
    // The pass was off for this run, so a report left by an earlier one no longer
    // describes the transforms that are actually baked.
    delete sample.metadata.physics;
  }
  meta.write(sample.dir, sample.metadata);
}

/** Rigid bake of resolved transforms plus the shared record. */
async function bakeAndRecord(sample, id, transforms, intent, physics) {
  const baked = await scene.bake(
    id,
    meta.ROLES.map((role) => ({ role, stem: sample.stem[role], trs: transforms[role] })),
    sample.dir,
  );
  record(sample, baked, intent, physics);
}

/**
 * Soft-body placement: the drape command solves, deforms and bakes in one step (the
 * intermediate is a vertex buffer, not a transform). A drape that throws falls back to
 * the rigid bake of the model's answer — same principle as every other physics failure.
 */
async function drapeAndRecord(sample, id, transforms, intent) {
  const draped = await scene.drape(
    id,
    {
      objects: meta.ROLES.map((role) => ({ role, stem: sample.stem[role], trs: transforms[role] })),
      options: PHYSICS_OPTIONS,
    },
    sample.dir,
  );
  record(sample, draped, intent, draped.report);
  return draped.report;
}

/** Returns its own log lines so that each sample prints as one block under concurrency. */
async function runSample(sourceDir, id, { dry, force }) {
  const sample = readSample(sourceDir, id);
  // A dry run costs nothing and bakes nothing, so it may re-print a sample that is already
  // placed — that is exactly how a finished corpus gets inspected. `--force` re-places one
  // for real; a local sample keeps its raw meshes, so that is repeatable.
  if (!dry && !force && meta.isPlaced(sample.metadata)) return [`▸ ${id}`, '  already placed'];

  // Meshes sitting in the sample folder itself (a local corpus) are voxelized in-process;
  // the usual volume-resident kind goes to the scene service.
  const grids = await scene.voxelize(
    id,
    meta.ROLES.map((role) => ({ role, stem: sample.stem[role], resolution: RESOLUTION[role] })),
    sample.dir,
  );
  const objects = Object.fromEntries(
    meta.ROLES.map((role) => [role, { name: sample.metadata[role].name, grid: grids[role], blocks: grids[role].blocks }]),
  );

  const { anchor, placed } = objects;
  const phrase = sample.metadata.placement;
  console.log(
    `⋯ ${id}  anchor ${anchor.grid.dims.join('x')} → ${anchor.blocks.length} block(s)` +
    `  placed ${placed.grid.dims.join('x')} → ${placed.blocks.length} block(s)`,
  );

  if (dry) return [`▸ ${id}`, buildPrompt({ phrase, a: anchor, b: placed })];

  // `requestPlacement` rejects an answer that fails validation, so a retry re-rolls a reply
  // that came back unusable rather than losing the sample until the next pipeline pass.
  const answer = await retry(() => requestPlacement({ phrase, a: anchor, b: placed }));
  const transforms = resolveTransforms(anchor, placed, answer);
  const intent = {
    contact: CONTACT_OVERRIDE ?? answer.contact,
    direction: answer.contact_direction,
    embed_fraction: answer.embed_fraction,
  };
  // The physics pass closes the gap between the voxel answer and the actual surfaces —
  // rigidly for most contacts, as a soft-body drape for cloth. Either way it refines or
  // it steps aside: any failure keeps the model's transforms and records why, because a
  // placement without refinement still beats a sample lost to a solver bug.
  const llmPlaced = transforms.placed;
  let physics = null;
  if (PHYSICS && intent.contact === 'drape') {
    try {
      physics = await drapeAndRecord(sample, id, transforms, intent);
      // A drape's ghost is the undeformed shape at the same pose — the pose never
      // changes, the geometry does — so the viewer is told to build it from the raw
      // mesh rather than a clone of the deformed one.
      if (process.env.PLACEMENT_GHOST === '1' && !physics.flags?.includes('drape_failed')) {
        console.log(`GHOST ${JSON.stringify({ ...transforms.placed, raw: true })}`);
      }
      logPlacement(id, phrase, answer, transforms, physics);
      return [
        `▸ ${id}  "${phrase}"`,
        `  scale=${round(answer.scale)}  yaw=${answer.yaw_degrees}  pitch=${answer.pitch_degrees}` +
        `  roll=${answer.roll_degrees}  center=[${rounded(answer.center)}]  contact=${contactNote(answer)}`,
        `  physics ${physicsSummary(physics)}`,
        `  ${answer.reasoning.replace(/\s+/g, ' ').slice(0, 220)}`,
      ];
    } catch (err) {
      physics = { contact: 'drape', error: err.message, flags: ['physics_failed'] };
    }
  } else if (PHYSICS) {
    try {
      const refined = await scene.refine(
        id,
        {
          objects: meta.ROLES.map((role) => ({ role, stem: sample.stem[role], trs: transforms[role] })),
          intent,
          options: PHYSICS_OPTIONS,
        },
        sample.dir,
      );
      physics = refined.report;
      transforms.placed = {
        position: rounded(refined.placed.position),
        rotation: rounded(refined.placed.rotation),
        scale: rounded(refined.placed.scale),
      };
    } catch (err) {
      physics = { contact: answer.contact, error: err.message, flags: ['physics_failed'] };
    }
  }

  // Logged before the bake, so an answer is kept even when what follows it fails.
  emitGhost(llmPlaced, transforms.placed);
  logPlacement(id, phrase, answer, transforms, physics);
  await bakeAndRecord(sample, id, transforms, intent, physics);

  return [
    `▸ ${id}  "${phrase}"`,
    `  scale=${round(answer.scale)}  yaw=${answer.yaw_degrees}  pitch=${answer.pitch_degrees}` +
    `  roll=${answer.roll_degrees}  center=[${rounded(answer.center)}]  contact=${contactNote(answer)}`,
    `  physics ${physics ? physicsSummary(physics) : 'off — baked the model\'s transforms as-is'}`,
    `  ${answer.reasoning.replace(/\s+/g, ' ').slice(0, 220)}`,
  ];
}

/** The model's contact, with an active override called out — a picker left on drape has
 *  surprised at least one person, so the log says exactly whose decision ran. */
function contactNote(answer) {
  if (!CONTACT_OVERRIDE || CONTACT_OVERRIDE === answer.contact) return answer.contact;
  return `${answer.contact} → OVERRIDDEN to ${CONTACT_OVERRIDE}`;
}

/**
 * Physics without a new model call: reads the pose the sample already carries out of its
 * posed GLBs, refines it against the true meshes under the intent recorded at placement
 * time, and re-bakes. What the viewer's "Apply Physics" button runs — and how a corpus
 * placed with `PLACEMENT_PHYSICS=off` gets its contacts closed afterwards. Unlike the
 * placing pass, a solver failure here fails the sample: physics was the whole request.
 */
async function refineSample(sourceDir, id) {
  const sample = readSample(sourceDir, id);
  if (!meta.isPlaced(sample.metadata)) {
    throw new Error('not placed yet — physics refines an existing placement');
  }

  const stems = meta.ROLES.map((role) => ({ role, stem: sample.stem[role] }));
  const { transforms } = await scene.pose(id, stems, sample.dir);
  // Samples placed before intents were recorded fall back to the previous report's
  // contact, then to attach — the universal "close the gap" refinement.
  const intent = sample.metadata.intent ?? {
    contact: sample.metadata.physics?.contact ?? 'attach',
    direction: [0, 0, 0],
    embed_fraction: 0,
  };
  if (CONTACT_OVERRIDE) intent.contact = CONTACT_OVERRIDE;

  // A drape re-pass re-drapes from the raw meshes under the same TRS — the deformation
  // lives in the file but the pose node is unchanged, so the transform read back is
  // exactly the one the original drape ran with.
  if (intent.contact === 'drape') {
    const physics = await drapeAndRecord(sample, id, transforms, intent);
    if (process.env.PLACEMENT_GHOST === '1' && !physics.flags?.includes('drape_failed')) {
      console.log(`GHOST ${JSON.stringify({ ...transforms.placed, raw: true })}`);
    }
    logPlacement(id, sample.metadata.placement, null, transforms, physics);
    return [
      `▸ ${id}  "${sample.metadata.placement}"`,
      ...(CONTACT_OVERRIDE ? [`  contact OVERRIDDEN to ${CONTACT_OVERRIDE}`] : []),
      `  physics ${physicsSummary(physics)}`,
    ];
  }

  const refined = await scene.refine(
    id,
    { objects: stems.map(({ role, stem }) => ({ role, stem, trs: transforms[role] })), intent, options: PHYSICS_OPTIONS },
    sample.dir,
  );
  const physics = refined.report;
  const baked = {
    anchor: transforms.anchor,
    placed: {
      position: rounded(refined.placed.position),
      rotation: rounded(refined.placed.rotation),
      scale: rounded(refined.placed.scale),
    },
  };

  emitGhost(transforms.placed, baked.placed);
  logPlacement(id, sample.metadata.placement, null, baked, physics);
  await bakeAndRecord(sample, id, baked, intent, physics);

  return [
    `▸ ${id}  "${sample.metadata.placement}"`,
    ...(CONTACT_OVERRIDE ? [`  contact OVERRIDDEN to ${CONTACT_OVERRIDE}`] : []),
    `  physics ${physicsSummary(physics)}`,
  ];
}

export async function placeSamples({ sourceDir, ids, dry = false, force = false, physicsOnly = false, concurrency = PLACE_WIDTH }) {
  // Each sample writes its own metadata before resolving, so a run cut short by rate limits
  // keeps everything that already landed and the next pass skips it.
  const results = await mapLimit(ids, dry ? 1 : concurrency, async (id) => {
    try {
      const lines = physicsOnly
        ? await refineSample(sourceDir, id)
        : await runSample(sourceDir, id, { dry, force });
      console.log(`\n${lines.join('\n')}`);
      return true;
    } catch (err) {
      console.error(`\n▸ ${id}\n  ✗ ${err.message}`);
      return false;
    }
  });

  const placed = results.filter(Boolean).length;
  const done = `${placed}/${ids.length}`;
  console.log(dry ? `\n${done} prompts built` : physicsOnly ? `\n${done} refined` : `\n${done} placed`);
  return ids.length - placed;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const requested = args.filter((arg) => !arg.startsWith('--'));
  const physicsOnly = args.includes('--physics-only');
  // Without both meshes there is nothing to place, and attempting one only reports that.
  // A named id is still tried, so asking about a specific sample still tells you why.
  // A physics-only pass wants placed samples instead — it continues from their poses.
  const ids = requested.length
    ? requested
    : meta
        .list(SOURCE_DIR)
        .filter((sample) => (physicsOnly ? meta.isPlaced(sample.metadata) : meta.isMeshed(sample.metadata)))
        .map((sample) => sample.id);
  const failed = await placeSamples({
    sourceDir: SOURCE_DIR,
    ids,
    dry: args.includes('--dry'),
    force: args.includes('--force'),
    physicsOnly,
  });
  if (failed) process.exitCode = 1;
}
