// Stage 3 on its own: hand every reference image that has no mesh yet to the Trellis farm.
//
// The opposite half in every way — GPU-bound, one campaign at a time, and the expensive
// part — which is exactly why it is worth starting separately, when the farm is free. It
// picks up whatever lacks a mesh, so it is safe to re-run and safe to run repeatedly as the
// image half keeps adding samples.
//
//   node pipeline/build-meshes.mjs

import { buildMeshes } from './build.mjs';

const ready = await buildMeshes();
if (ready.length) console.log('Place them with:  node pipeline/run.mjs');
