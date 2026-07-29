// The placement half on its own: voxelize the meshes a generation run already made,
// ask every model under test where the object goes, and bake each answer into its
// own copy of the GLB. Nothing here touches a GPU or an image model, so it is cheap
// to re-run as the model list changes.
//
//   node pipeline/test-place.mjs [sampleId ...] [--models=a,b] [--force]

import { placeSamples, sampleIds } from './test-placement.mjs';

const failed = await placeSamples(sampleIds());
if (failed) process.exitCode = 1;
