// What a given elevation and azimuth actually look at.
//
// The angles handed to /render are in the mask's own frame — +Y up, +Z the model's front —
// and modal/voxhammer_render.py turns them into Blender's before bpy-renderer ever sees
// them. That turn is worth measuring against a real asset rather than trusting, so this
// renders one already-masked sample across a sweep of angles and prints how much of the
// mask each one caught. It talks to the service directly rather than through
// voxhammer.mjs, because it wants the whole coverage table and not the chosen view.
//
//   node pipeline/probe-camera.mjs sample-4 --elevations=15,30,45,60,75,85
//   node pipeline/probe-camera.mjs sample-1 --elevations=30 --azimuths=0,90,180,270

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ENV_FILE = path.join(ROOT, '.env');
if (fs.existsSync(ENV_FILE)) process.loadEnvFile(ENV_FILE);

const BASE_URL = process.env.VOXHAMMER_BASE_URL ?? 'https://starshot-aitools--dc-voxhammer-web.modal.run';

const args = process.argv.slice(2);
const flag = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const numbers = (name, fallback) => flag(name, fallback).split(',').map(Number);

const id = args.find((a) => !a.startsWith('--'));
if (!id) throw new Error('usage: node pipeline/probe-camera.mjs <sample> [--elevations=..] [--azimuths=..]');

const dir = path.join(path.resolve(ROOT, flag('out', 'edit-results')), id);
const elevations = numbers('elevations', '15,30,45,60,75');
const azimuths = numbers('azimuths', '0');
const cameras = elevations.flatMap((elevation) => azimuths.map((azimuth) => ({ elevation, azimuth })));

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const form = new FormData();
form.append('model', new Blob([fs.readFileSync(path.join(dir, 'model.glb'))]), 'model.glb');
form.append('mask', new Blob([fs.readFileSync(path.join(dir, 'mask.glb'))]), 'mask.glb');
form.append('cameras', JSON.stringify(cameras));
form.append('sample', `probe-${id}`);

console.log(`${id} → ${cameras.length} angle(s)`);
const spawn = await fetch(`${BASE_URL}/render`, { method: 'POST', body: form, signal: AbortSignal.timeout(600_000) });
if (!spawn.ok) throw new Error(`POST /render ${spawn.status}: ${(await spawn.text()).slice(0, 300)}`);
const { job_id: jobId } = await spawn.json();
console.log(`job ${jobId}`);

let status;
for (let stage = null; ; ) {
  await sleep(10_000);
  const response = await fetch(`${BASE_URL}/jobs/${jobId}`, { signal: AbortSignal.timeout(30_000) });
  if (!response.ok) continue;
  status = await response.json();
  if (status.stage !== stage) console.log(`  ${(stage = status.stage)}`);
  if (status.status === 'done' || status.status === 'failed') break;
}

console.log('');
for (const { elevation, azimuth, coverage } of status.cameras_seen ?? []) {
  const shown = coverage === null ? '(not rendered)' : `${(coverage * 100).toFixed(3)}%`;
  console.log(`  elev ${String(elevation).padStart(5)}   azim ${String(azimuth).padStart(5)}   ${shown.padStart(14)}`);
}

if (status.status === 'done') {
  console.log(`\nchosen: elev ${status.view.elevation}  azim ${status.view.azimuth}`);
  const png = await fetch(`${BASE_URL}/jobs/${jobId}/file/${encodeURIComponent('images/2d_render.png')}`);
  fs.writeFileSync(path.join(dir, 'probe.png'), Buffer.from(await png.arrayBuffer()));
  console.log(`view → ${path.relative(ROOT, path.join(dir, 'probe.png'))}`);
} else {
  console.log(`\n${status.error}`);
}
