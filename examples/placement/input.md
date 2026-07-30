# Placement — user prompt (full example)

The user message for one sample, exactly as `buildPrompt` in `pipeline/place.mjs` assembles
it: the placement phrase, then each object as its voxel occupancy (default resolution 16
along the longest axis) decomposed into disjoint maximal axis-aligned blocks. Blocks are
printed in the object's own bbox-centred frame in world units, largest first, so the
load-bearing structure comes before the detail. It is sent together with the system message
in [system.md](system.md).

This is a real published sample — `freestanding_bathtub_439b202c` with its bath brush,
phrase written in the `bare` phrasing (names neither object) — voxelized natively from the
local corpus by `node pipeline/run.mjs --dry --source=<dir> <id>`. At resolution 16 the tub
is 44 blocks and the brush 8, about 3.5 KB in total: the first block is the tub's floor,
the next four are its walls, and the basin is the empty space they enclose. The brush's
first block is the handle shaft, with the bristle head widening it at the -x end.

```
PLACEMENT: "leaning against the outside front face, slightly biased to the left"

A (anchor, stays fixed) — "freestanding bathtub"
bounding box 1.0069 x 0.3613 x 0.551  (x, y, z), centred at the origin — 44 block(s), largest first
(-0.315, -0.189, -0.22) to (0.315, -0.063, 0.22)
(-0.378, -0.063, -0.22) to (0.378, 0.189, -0.157)
(-0.378, -0.063, 0.157) to (0.378, 0.189, 0.22)
(-0.252, 0, -0.283) to (0.252, 0.189, -0.22)
(-0.252, 0, 0.22) to (0.252, 0.189, 0.283)
(-0.378, -0.189, -0.157) to (-0.315, 0, 0.157)
(-0.441, -0.126, -0.094) to (-0.378, 0.189, 0.094)
(0.315, -0.189, -0.157) to (0.378, 0, 0.157)
(0.378, 0.063, -0.22) to (0.441, 0.189, 0.22)
(-0.315, -0.063, -0.157) to (0.315, 0, -0.094)
(-0.315, -0.063, 0.094) to (0.315, 0, 0.157)
(0.441, 0.063, -0.094) to (0.503, 0.189, 0.094)
(0.378, -0.063, -0.094) to (0.441, 0.063, 0.094)
(-0.503, 0.063, -0.094) to (-0.441, 0.189, 0.094)
(-0.441, 0.063, -0.22) to (-0.378, 0.189, -0.094)
(-0.441, -0.063, 0.094) to (-0.378, 0.189, 0.157)
(0.252, 0, 0.094) to (0.441, 0.063, 0.157)
(0.252, 0, -0.157) to (0.441, 0.063, -0.094)
(-0.441, 0, -0.157) to (-0.252, 0.063, -0.094)
(0.378, -0.189, -0.031) to (0.441, -0.063, 0.031)
(-0.378, 0, 0.031) to (-0.315, 0.063, 0.157)
(-0.378, 0.126, -0.283) to (-0.252, 0.189, -0.22)
(0.252, 0.063, -0.283) to (0.315, 0.189, -0.22)
(-0.441, 0.063, 0.157) to (-0.378, 0.189, 0.22)
(-0.378, 0.126, 0.22) to (-0.252, 0.189, 0.283)
(0.252, 0.063, 0.22) to (0.315, 0.189, 0.283)
(-0.315, 0.063, -0.283) to (-0.252, 0.126, -0.22)
(-0.315, 0.063, 0.22) to (-0.252, 0.126, 0.283)
(0.315, 0, -0.094) to (0.378, 0.063, -0.031)
(0.315, 0.063, 0.094) to (0.378, 0.126, 0.157)
(-0.503, 0.126, 0.094) to (-0.441, 0.189, 0.157)
(0.441, 0.126, 0.094) to (0.503, 0.189, 0.157)
(-0.378, 0.063, -0.157) to (-0.315, 0.126, -0.094)
(0.315, 0.126, -0.283) to (0.378, 0.189, -0.22)
(0.441, 0.126, -0.157) to (0.503, 0.189, -0.094)
(-0.441, -0.189, -0.031) to (-0.378, -0.126, 0.031)
(0.378, -0.126, 0.031) to (0.441, -0.063, 0.094)
(-0.378, 0, -0.094) to (-0.315, 0.063, -0.031)
(0.315, 0.063, -0.157) to (0.378, 0.126, -0.094)
(0.315, 0, 0.031) to (0.378, 0.063, 0.094)
(-0.378, 0.063, 0.094) to (-0.315, 0.126, 0.157)
(-0.503, 0.126, -0.157) to (-0.441, 0.189, -0.094)
(0.315, 0.126, 0.22) to (0.378, 0.189, 0.283)
(-0.315, 0, 0.094) to (-0.252, 0.063, 0.157)

B (place this) — "bath brush"
bounding box 1.0057 x 0.1235 x 0.2555  (x, y, z), centred at the origin — 8 block(s), largest first
(-0.503, -0.063, -0.094) to (0.44, 0, 0.094)
(-0.503, 0, -0.094) to (-0.189, 0.063, 0.094)
(-0.189, 0, -0.031) to (0.503, 0.063, 0.031)
(-0.44, -0.063, 0.094) to (-0.314, 0.063, 0.157)
(-0.44, -0.063, -0.157) to (-0.314, 0.063, -0.094)
(0.44, -0.063, -0.031) to (0.503, 0, 0.031)
(-0.314, -0.063, -0.157) to (-0.251, 0, -0.094)
(-0.314, -0.063, 0.094) to (-0.251, 0, 0.157)
```
