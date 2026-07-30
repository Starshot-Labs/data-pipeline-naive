// Placement physics off the main thread. The refine step is the one CPU-bound piece of
// stage 5 that runs in this process (remote mode ships it to scene-ops, which fans out
// across containers on its own), and the placement stage runs hundreds of samples wide —
// a half-second solve on the event loop would serialize all of them and starve the
// in-flight model calls. A small pool of workers keeps refinement as parallel as the
// machine allows while the main thread stays free to shepherd HTTP.
//
// Workers are unref'd when idle and ref'd while working, so an exiting pipeline never
// hangs on the pool and a pending job never lets the process quit early.

import os from 'node:os';
import { Worker } from 'node:worker_threads';

const poolSize = () => {
  const asked = Number(process.env.PHYSICS_WORKERS ?? 0);
  const cores = os.availableParallelism?.() ?? os.cpus().length;
  return Math.max(1, Math.min(asked > 0 ? asked : cores - 1, 16));
};

let slots = null;
const queue = [];
const pending = new Map();
let nextId = 1;

function spawn() {
  const slot = { worker: null, busy: false, current: 0 };
  slot.worker = new Worker(new URL('./physics-worker.mjs', import.meta.url));

  slot.worker.on('message', ({ id, result, error }) => {
    const job = pending.get(id);
    pending.delete(id);
    slot.busy = false;
    slot.current = 0;
    slot.worker.unref();
    if (job) error ? job.reject(new Error(error)) : job.resolve(result);
    drain();
  });

  // A crashed worker takes its current job down with it; the slot gets a fresh worker so
  // one bad mesh cannot shrink the pool for the rest of the run.
  slot.worker.on('error', (err) => {
    const job = pending.get(slot.current);
    pending.delete(slot.current);
    const at = slots.indexOf(slot);
    if (at >= 0) slots[at] = spawn();
    if (job) job.reject(err);
    drain();
  });

  slot.worker.unref();
  return slot;
}

function drain() {
  while (queue.length) {
    const slot = slots.find((s) => !s.busy);
    if (!slot) return;
    const job = queue.shift();
    pending.set(job.id, job);
    slot.busy = true;
    slot.current = job.id;
    slot.worker.ref();
    slot.worker.postMessage({ id: job.id, job: job.job });
  }
}

function submit(job) {
  slots ??= Array.from({ length: poolSize() }, spawn);
  return new Promise((resolve, reject) => {
    queue.push({ id: nextId++, job, resolve, reject });
    drain();
  });
}

/** Run `refineDir(dir, payload)` on a pooled worker. */
export const refine = (dir, payload) => submit({ op: 'refine', dir, payload });

/** Run `drapeDir(source, destination, …)` on a pooled worker — solve, deform and bake. */
export const drape = (source, destination, payload) => submit({ op: 'drape', source, destination, payload });
