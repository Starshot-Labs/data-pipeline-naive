// Step 3: image-to-3D on the batched Trellis 2 farm.
//
// Nothing here is batched. Every object goes into one campaign, at any size, so they all
// generate simultaneously across the farm's GPU pools.
//
// The campaign runs two lanes. The farm reads the *folder*, not a parameter: whatever lands
// in `texture_dir` gets a UV unwrap and PBR bake, whatever lands in `geometry_dir` does not.
// Anchors take the geometry lane — they only have to be voxelized and reasoned against — and
// placed objects take the texture lane, since those are the ones that show. Both lanes go in
// a single campaign on purpose: phase 1 is bit-identical between the modes, so splitting
// them would pay twice for the input-volume walk, the reservation and the warm.
//
// That means the images do *not* go through `POST /upload`: an HTTP body is capped at 4 GiB
// and a request at 150 s, which at roughly a megabyte an image tops out near 3,500 objects
// and would force exactly the slicing we do not want. `scene.stageImages` puts them on the
// input Volume instead — a copy when the pipeline is already on Modal, the Modal CLI when it
// is not — and `/run` then reads those folders with no files attached.
//
// A new campaign cancels whatever is already on the farm with `POST /reset` before
// uploading — campaigns cannot overlap, and waiting them out left us blocked behind
// stale runs that would never produce anything we still wanted.
//
// The meshes never come back here. `dc-scene-ops` collects them straight off the farm's
// output volume onto ours, along with the images that produced them, which is a volume-to-
// volume copy rather than a download and a re-upload.
//
// `jobs` holds one entry per mesh still missing:
//   { id, role: 'anchor' | 'placed', stem: '<slug>_<uuid>',
//     image: '<absolute path to the local reference image>' }
//
// Failures are logged rather than thrown. Whatever did not land stays unrecorded, so
// re-running `build.mjs` retries exactly that.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as scene from './scene.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const DEFAULT_BASE_URL = 'https://starshot-aitools--trellis2-latent-farm-web.modal.run';
// The extensions the farm's dispatcher picks up; anything else is silently ignored there.
const ACCEPTED = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp']);

const LANE_OF_ROLE = { anchor: 'geo', placed: 'tex' };

// Read lazily: this module is evaluated before the entry point loads .env.
const baseUrl = () => process.env.TRELLIS_BASE_URL ?? DEFAULT_BASE_URL;
const pollMs = () => Number(process.env.TRELLIS_POLL_S ?? 10) * 1000;

// Deliberately far past anything the farm can actually take: a campaign call is capped at
// 24 h server-side, so that limit should always be what fires, never a client timeout.
const campaignTimeoutMs = () => Number(process.env.TRELLIS_RUN_TIMEOUT_S ?? 172_800) * 1000;
const requestTimeoutMs = () => Number(process.env.TRELLIS_REQUEST_TIMEOUT_S ?? 600) * 1000;
// `campaign_describe` walks the whole input Volume before publishing any state, and gets
// slower as the corpus grows, so /report can legitimately 404 for a long while at first.
const publishGraceMs = () => Number(process.env.TRELLIS_PUBLISH_GRACE_S ?? 3_600) * 1000;
// Once it has published, a 404 means the state went away rather than that it never arrived,
// so only ride out something as short as a redeploy before calling the campaign lost.
const vanishGraceMs = () => Number(process.env.TRELLIS_VANISH_GRACE_S ?? 120) * 1000;
// The farm drops objects whose decoder mesh blows past its face cap, so the GLB count can
// legitimately stop short of the whole campaign. Once it stops moving, that is the answer.
const assetSettleMs = () => Number(process.env.TRELLIS_ASSET_SETTLE_S ?? 180) * 1000;

const stateFile = () => path.resolve(ROOT, process.env.TRELLIS_STATE ?? '.trellis-campaign.json');

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const refOf = (job) => `${job.id}/${job.stem}`;

/**
 * The key→job mapping of an unfinished campaign, so a crashed or interrupted run can attach
 * to it instead of paying for the work twice. The farm's manifest is keyed
 * `campaign:<run_id>:<object_id>`, so reusing the run id also means a re-POST would only
 * pick up whatever did not finish.
 */
function saveCampaign(runId, keyed) {
  const keys = Object.fromEntries([...keyed].map(([key, job]) => [key, refOf(job)]));
  fs.writeFileSync(stateFile(), JSON.stringify({ run_id: runId, created_at: new Date().toISOString(), keys }, null, 2));
}

function loadCampaign(jobs) {
  if (!fs.existsSync(stateFile())) return null;

  const state = JSON.parse(fs.readFileSync(stateFile(), 'utf8'));
  const byRef = new Map(jobs.map((job) => [refOf(job), job]));
  const keyed = new Map();
  for (const [key, ref] of Object.entries(state.keys ?? {})) {
    const job = byRef.get(ref);
    if (job) keyed.set(key, job);
  }
  return keyed.size ? { runId: state.run_id, keyed } : null;
}

const clearCampaign = () => fs.rmSync(stateFile(), { force: true });

async function api(pathname, { method = 'GET', body, params = {}, timeoutMs } = {}) {
  const url = new URL(baseUrl() + pathname);
  for (const [key, value] of Object.entries(params)) url.searchParams.set(key, String(value));
  return fetch(url, { method, body, signal: AbortSignal.timeout(timeoutMs ?? requestTimeoutMs()) });
}

async function detail(response, label) {
  return new Error(`${label} ${response.status}: ${(await response.text()).slice(0, 200)}`);
}

/**
 * Cancel whatever the farm is currently running. A new campaign's `/run` would clear the
 * queues on its own, but an explicit `/reset` drops the old epoch immediately and frees
 * the local checkpoint that would otherwise try to attach to a run we no longer want.
 */
async function cancelActiveCampaign() {
  const response = await api('/status');
  if (!response.ok) throw await detail(response, 'GET /status');

  const { dispatch } = await response.json();
  if (!dispatch?.active) {
    clearCampaign();
    return;
  }

  console.log(`    cancelling campaign "${dispatch.run_id}"`);
  const reset = await api('/reset', { method: 'POST' });
  if (!reset.ok) throw await detail(reset, 'POST /reset');
  clearCampaign();

  // /reset is async on the farm — wait until it reports idle before we /run on top of it.
  const deadline = Date.now() + requestTimeoutMs();
  while (Date.now() < deadline) {
    await sleep(pollMs());
    const status = await api('/status');
    if (!status.ok) throw await detail(status, 'GET /status');
    if (!(await status.json()).dispatch?.active) return;
  }
  throw new Error(`campaign "${dispatch.run_id}" did not stop after /reset`);
}

/**
 * One staging entry per image, keyed `o<n>`.
 *
 * The renaming is deliberate: the farm derives each object id by slugifying the file stem, so
 * handing it names that are already slug-safe makes the id map straight back to a job instead
 * of having to reverse the slug. Keys stay unique across both lanes, because the farm keeps
 * only the stem in an object id and restarting the numbering per lane would map two assets
 * onto one job.
 */
function stagingObjects(keyed) {
  const objects = [];
  for (const [key, job] of keyed) {
    const extension = path.extname(job.image).toLowerCase();
    if (!ACCEPTED.has(extension)) {
      console.error(`    ✗ ${job.id} [${job.role}]: the farm ignores ${extension} images`);
      continue;
    }
    const lane = LANE_OF_ROLE[job.role];
    if (!lane) throw new Error(`no lane defined for role "${job.role}"`);
    objects.push({ source: job.image, key, lane });
  }
  return objects;
}

async function startCampaign(runId, dirs) {
  // Volume paths rather than bare folder names: with no files attached the farm reads these
  // from the volume root, and a bare name matches nothing at all. No `texture_mode` either —
  // the lanes already name their mode, and sending both is a 400.
  const response = await api('/run', {
    method: 'POST',
    params: { run_id: runId, mode: 'finish', ...dirs },
  });
  if (!response.ok) throw await detail(response, 'POST /run');

  const result = await response.json();
  if (!result.started) throw new Error(`campaign did not start: ${JSON.stringify(result).slice(0, 200)}`);
}

/** Before giving up on a campaign, take whatever it managed to write. */
async function salvage(runId, expected, reason) {
  const landed = await scene.assetCount(runId);
  if (!landed) throw Object.assign(new Error(reason), { terminal: true });
  console.log(`    ${reason} — ${landed}/${expected} asset(s) landed, collecting those`);
  return {};
}

/**
 * Waits for the campaign's GLBs, not for the farm to agree that it is finished.
 *
 * The assets are the only thing we came for, so they decide when this is over. `/report`
 * is watched too, since it carries the cost figures and turns 200 first in the happy case,
 * but it cannot be the sole trigger: a dispatcher that dies after writing the meshes, or
 * one that simply never publishes a final state, would otherwise strand a run that had
 * already produced everything. Counting them is a directory listing on the mounted volume
 * rather than a CLI subprocess, so it happens on every poll.
 */
async function awaitCampaign(runId, expected) {
  const deadline = Date.now() + campaignTimeoutMs();
  let missingSince = null;
  let published = false;
  let lastState = null;
  let lastDone = -1;
  let assets = 0;
  let assetsSince = Date.now();

  while (Date.now() < deadline) {
    await sleep(pollMs());

    const landed = await scene.assetCount(runId);
    if (landed !== assets) {
      assets = landed;
      assetsSince = Date.now();
    }
    if (assets >= expected) return {};
    if (assets > 0 && Date.now() - assetsSince > assetSettleMs()) {
      console.log(`    ${assets}/${expected} assets written and no longer growing — collecting those`);
      return {};
    }

    const response = await api('/report', { params: { run_id: runId } });

    // Before the first report a 404 is ordinary — a campaign that raised before publishing
    // looks exactly like a describe still walking the volume, so it gets a long grace and
    // stays attachable, since it may well still be running. Afterwards a 404 cannot mean
    // that any more: the state existed and is gone, which is the farm going away underneath
    // us. That one is terminal, because re-attaching to it would only 404 forever.
    if (response.status === 404) {
      missingSince ??= Date.now();
      if (published) {
        if (Date.now() - missingSince > vanishGraceMs()) {
          return salvage(runId, expected, 'campaign state vanished — the farm stopped serving it');
        }
      } else if (Date.now() - missingSince > publishGraceMs()) {
        throw new Error('campaign never published a state');
      }
      continue;
    }
    missingSince = null;
    published = true;

    const body = await response.json();
    // Only 200 means finished. 202 is "still running", and `response.ok` covers both.
    if (response.status === 200) return body.report ?? {};
    if (body.state === 'error' || body.state === 'empty') {
      return salvage(runId, expected, `campaign ended ${body.state}`);
    }

    // A report can arrive with no `dispatch` block at all, which is not the same as no
    // progress. Reading that as zero is what printed `0/?` and then latched `lastDone` on
    // it, hiding every later update behind a count that never appeared to change.
    const done = body.dispatch?.objects_done ?? null;
    if (body.state !== lastState || (done !== null && done !== lastDone)) {
      lastState = body.state;
      if (done !== null) lastDone = done;
      const progress = done === null ? '' : ` ${done}/${body.dispatch.objects_total ?? '?'} objects`;
      const eta = body.dispatch?.eta ? `, eta ${Math.round(body.dispatch.eta)}s` : '';
      console.log(`    ${body.state}${progress}${eta}${assets ? `, ${assets} asset(s) written` : ''}`);
    }
  }
  throw new Error(`campaign unfinished after ${campaignTimeoutMs() / 1000}s`);
}

// Ways the farm gives up on an object for good. Everything else it fails to deliver — a dead
// shard, a transient error — is worth handing back to the next campaign.
const PERMANENT_LOSS = ['dropped_oversize', 'dropped_degenerate', 'dropped_oom', 'geom_failures'];

/**
 * The jobs this campaign will never deliver, however many times it is re-run.
 *
 * The report counts its drops but never names them, so the objects themselves are whatever
 * the collect step could not find. That only tells the two cases apart when every one of
 * them is accounted for by a permanent cause: an oversize mesh is deterministic and will
 * blow past the face cap again, whereas anything lost transiently deserves another go. A
 * campaign that lost even one object the retryable way therefore abandons nothing.
 */
function abandonedJobs(report, missing) {
  const permanent = PERMANENT_LOSS.reduce((total, key) => total + (report.phase2?.[key] ?? 0), 0);
  return permanent > 0 && missing.length === permanent ? missing : [];
}

/**
 * Builds every job in a single campaign, then has `dc-scene-ops` move the results onto our
 * volume. Failures are logged, not thrown; the return says which meshes landed and which
 * jobs the farm gave up on for good.
 */
export async function generateMeshes(jobs) {
  const [farm, ops] = await Promise.all([
    api('/health').then((response) => (response.ok ? 'healthy' : `HTTP ${response.status}`), (err) => err.message),
    scene.health(),
  ]);
  console.log(`  ${baseUrl()} → ${farm}`);
  console.log(`  ${scene.url()} → ${ops}`);

  // Attach to a campaign we already started rather than starting a second one, which
  // would cancel it and redo work the farm has already billed us for.
  const previous = loadCampaign(jobs);
  const attached = previous && (await api('/report', { params: { run_id: previous.runId } })).status !== 404
    ? previous
    : null;

  const runId = attached?.runId ?? `dc-${Date.now().toString(36)}`;
  const keyed = attached?.keyed ?? new Map(jobs.map((job, i) => [`o${i}`, job]));
  console.log(
    attached
      ? `  ▸ ${runId}  attaching to campaign already in flight for ${keyed.size} object(s)`
      : `  ▸ ${runId}  ${jobs.length} object(s), all in one campaign`,
  );

  const started = Date.now();
  try {
    if (!attached) {
      await cancelActiveCampaign();
      const { dirs, counts } = await scene.stageImages(runId, stagingObjects(keyed));
      console.log(`    staged ${counts.tex ?? 0} textured + ${counts.geo ?? 0} geometry-only image(s)`);
      saveCampaign(runId, keyed);
      await startCampaign(runId, dirs);
    }
    const report = await awaitCampaign(runId, keyed.size);

    const { collected, missing, images } = await scene.collect(
      runId,
      [...keyed].map(([key, job]) => ({ sample: job.id, role: job.role, stem: job.stem, key, lane: LANE_OF_ROLE[job.role] })),
    );

    // The centred crop Trellis reconstructs from is the one that matches the mesh, so a run
    // that falls back to the photo we sent says so rather than quietly mixing the two.
    if (images?.reference) {
      console.log(`    ⚠ ${images.reference} image(s) fell back to the uncentred reference — no images/ in this run`);
    }
    if (images?.none) console.log(`    ⚠ ${images.none} object(s) landed with no image at all`);

    const gpu = report.gpu_s_per_object?.total;
    const dropped = report.phase2?.dropped_oversize ?? 0;
    const minutes = ((Date.now() - started) / 60_000).toFixed(1);
    console.log(
      `    ✓ ${collected.length}/${jobs.length} meshes in ${minutes} min` +
      `${dropped ? `, ${dropped} dropped oversize` : ''}${gpu ? `, ${gpu.toFixed(1)} GPU-s/object` : ''}`,
    );
    clearCampaign();
    return { collected, abandoned: abandonedJobs(report, missing) };
  } catch (err) {
    console.error(`    ✗ ${runId}: ${err.message}`);
    // Keep the state only while the campaign might still be alive; a campaign that ended
    // in error or empty will never produce anything, so re-attaching would loop forever.
    if (err.terminal) clearCampaign();
    return { collected: [], abandoned: [] };
  }
}
