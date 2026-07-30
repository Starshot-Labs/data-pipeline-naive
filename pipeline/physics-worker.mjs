// Worker half of physics-pool.mjs: one solver job in, one result or error out. ops.mjs
// is imported only when a drape job arrives, so refine-only runs never load sharp.

import { parentPort } from 'node:worker_threads';
import { refineDir } from './physics.mjs';

parentPort.on('message', async ({ id, job }) => {
  try {
    const result =
      job.op === 'drape'
        ? (await import('./ops.mjs')).drapeDir(job.source, job.destination, job.payload.objects, job.payload.options)
        : refineDir(job.dir, job.payload);
    parentPort.postMessage({ id, result });
  } catch (err) {
    parentPort.postMessage({ id, error: err?.stack ?? String(err) });
  }
});
