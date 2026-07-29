// The generation half on its own: an isometric shot of each anchor, nano banana's
// re-posed photo of the object, and the mesh Trellis builds from it. Model-agnostic
// and the expensive half, so it is worth running once and then benchmarking as many
// models as you like against it with pipeline/test-place.mjs.
//
//   node pipeline/test-generate.mjs [sampleId ...] [--force]

import { generateSamples, sampleIds } from './test-placement.mjs';

const ready = await generateSamples(sampleIds());
if (ready.length) console.log('Place them with:  node pipeline/test-place.mjs');
