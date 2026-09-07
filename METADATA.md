# `generated/<id>/metadata.json`

One file per sample, covering the whole pipeline. Replaces `spec.json` and
`out/<id>/result.json`.

```json
{
  "id": "kitchen_island_f29f9f7d-02ff-4747-b83a-0d42ef995fb8",
  "uuid": "f29f9f7d-02ff-4747-b83a-0d42ef995fb8",
  "created_at": "2026-07-28T00:40:50.607Z",
  "context": "coffee shop",
  "category": "rigid",
  "complexity": "simple",
  "relation": "on top of",
  "detail": "position",
  "placement": "cup on top of the kitchen island in the middle",
  "anchor": {
    "name": "kitchen island",
    "description": "A long rectangular kitchen island in pale oak, with a thick stone top and a bank of drawers down one side.",
    "objaverse": {
      "uid": "8476c4170df24cf5bbe6967222d1a42d",
      "glb": "glbs/000-023/8476c4170df24cf5bbe6967222d1a42d.glb",
      "license": "by"
    },
    "image": "kitchen_island_f29f9f7d-02ff-4747-b83a-0d42ef995fb8.png",
    "mesh": "kitchen_island_f29f9f7d-02ff-4747-b83a-0d42ef995fb8.glb",
    "textured": true,
    "size": [1.823244, 0.941002, 0.86134]
  },
  "placed": {
    "name": "cup",
    "description": "A small white ceramic cup with a rounded body and a single looping handle.",
    "objaverse": {
      "uid": "1d0f4e7c2a1b4a0f9c3e5d6b7a8c9e0f",
      "glb": "glbs/000-104/1d0f4e7c2a1b4a0f9c3e5d6b7a8c9e0f.glb",
      "license": "by-sa"
    },
    "image": "cup_f29f9f7d-02ff-4747-b83a-0d42ef995fb8.png",
    "mesh": "cup_f29f9f7d-02ff-4747-b83a-0d42ef995fb8.glb",
    "textured": true,
    "size": [0.112806, 0.09004, 0.0901]
  },
  "combined_size": [1.823244, 1.031042, 0.86134],
  "intent": {
    "contact": "rest",
    "direction": [0, 1, 0],
    "embed_fraction": 0
  },
  "physics": {
    "contact": "rest",
    "moved": 0.004182,
    "rotated_degrees": 0,
    "gap_before": 0.0121,
    "gap_after": -0.0008,
    "flags": []
  }
}
```

Number arrays are folded onto one line above to keep the example readable; the real file is
`JSON.stringify(…, null, 2)`, which puts every element on its own.

`placement.txt` stays alongside it holding the same four lines as `placement`, one per line
with a trailing newline. Both are written and must match.

A finished sample lives at `trellis-scene-vol-v2:datasets/raw/stage1/<id>/`, assembled there
by two writers rather than uploaded whole. `dc-scene-ops` writes the posed meshes and their
reference images straight off the volumes; `pipeline/upload.mjs` adds metadata.json and
placement.txt afterwards. Metadata lands **last**, and its presence marks a sample done.

Locally a sample folder holds only the small half. The meshes never come down — `mesh` names
the file as it exists on the volume.

## Fields

- `id` — folder name, `<anchor slug>_<uuid>`
- `uuid` — kept even though `id` ends with it and every filename contains it. Deliberate.
- `created_at` — ISO timestamp from stage 1
- `context` — one of the entries in `CONTEXTS`
- `pair` — backfill samples only: `<anchor uid>|<placed uid>`, the couple the sample re-poses.
  Derivable from the two `objaverse` blocks, but recorded so grouping needs no join — once
  pairs repeat, train/val splits have to split by pair, or validation leaks the very
  memorisation the backfill exists to break. Its presence is what marks a backfill sample.
- `tier` — backfill samples only: `natural` when both the planner's pool-tag test and the
  writer model called the pairing natural for its category, `stretch` when either loosened —
  the filter to reach for when a training run wants only everyday scenes
- `category` — which of the six placement categories in `scene-spec.mjs` the sample was dealt:
  `rigid`, `soft`, `penetrative`, `containment`, `bonded` or `noncontact`
- `complexity` — `simple` or `complex`, the half of the category's relation list it drew from
- `relation` — the spatial primitive it was dealt, e.g. `on top of`, `inside`, `laid over`
- `detail` — how much the phrase says beyond the two names and the relation: `bare` (nothing),
  `position` (one plain direction of the model's choosing), or `part` (tied to a real
  geometric part of the anchor)
- `placement` — the instruction phrase in four progressively shortened forms: as written,
  then without the anchor named, without the placed object, and without either. The point is
  that a model sees the same pose described at four levels of grounding, so it cannot lean on
  the object names alone. Samples written before the forms existed, and the handful whose
  names overlap too much for a clean deletion, carry a single string here instead — both
  shapes are valid on disk, which is why nothing reads this field directly.
- `placement_original` — the first form on its own. Redundant by construction and kept
  anyway: it is what the placement prompt, the dedup keys and the logs all want, and reading
  it is what stops a four-line array reaching code that expected a sentence. Use
  `metadata.mjs`'s `phraseOf` / `variantsOf` / `placementText` rather than either field.
- `anchor` / `placed` — one block each, identical shape:
    - `name` — readable name (`"kitchen island"`), never the slug
    - `description` — one-sentence visual description, faithful to the seeded asset
    - `objaverse` — the seed the sample was born with: `uid`, the `glb` path within the
      Hugging Face dataset, and the `license` where Objaverse records one
    - `image` / `mesh` — filenames including extension
    - `textured` — whether the mesh carries its own materials
    - `size` — `[x, y, z]` of the baked mesh
- `combined_size` — `[x, y, z]` union box of the two posed meshes
- `intent` — the placement model's classification of its own answer (`contact` —
  `rest`, `lean`, `attach`, `embed`, `drape` or `none` — plus `contact_direction` as
  `direction` and `embed_fraction`), recorded so a physics-only re-pass
  (`run.mjs --physics-only`, the viewer's "Apply Physics") knows what the placement
  meant without another model call. A `drape` bakes deformed vertices rather than a new
  transform; the placement TRS in the GLB stays the model's.
- `physics` — what the mesh-contact refinement between place and bake did:
    - `contact` — the intent the solver acted on, which picked its behaviour
    - `moved` / `rotated_degrees` — how far the pass moved the placed object, in world
      units and degrees
    - `gap_before` / `gap_after` — closest approach to the anchor before and after
      (negative means touching with the intended sliver of overlap, or an embed)
    - `flags` — empty when the solve was clean; `deep_penetration`, `no_contact`,
      `drifted`, `settled_away` (the surface would have slid the body away from where
      the phrase put it — or toppled it outright — so the straight drop was kept
      instead), `nested` (wedged or
      tucked somewhere tighter than the field resolves — the pose stands, lifted at most
      a few cells to its least-penetrating height), `dropped_far`,
      `no_rest`, `lean_no_side`, `lean_incomplete`, `no_direction`, `drape_failed` (the
      cloth pass refused and the rigid bake of the model's answer stands),
      `drape_budget` (the cloth ran out of its wall-clock budget — the pose is a real
      partial drape in contact, not a failure, but it had not finished settling) mean the
      pass reverted, held something back, or wants review; `error` (with no numbers) means
      it crashed and the model's transforms were baked untouched

Keys appear as the stage that owns them completes. Stage 1 writes everything down to the
descriptions and the seeds; the fetch adds `image`, `mesh` and `textured`; the bake adds the
sizes, `intent` and `physics`. Omit a key that is not ready rather than writing `null` —
absence is how the pipeline knows what work is left. `physics` is absent when
`PLACEMENT_PHYSICS=off` — deliberately deleted then, because a report left by an earlier pass
would describe transforms that are no longer the baked ones.

## Key order

The order above is the order on disk, top level and inside a role block alike. It is not
whatever order the stages happened to assign in: `pipeline/metadata.mjs` owns the list and
every write goes through it, so a sample reads the same whether it was placed in one pass or
re-placed a week later, and the published copy is byte-for-byte the local one. A key the list
does not know is kept and written last rather than dropped.

## Dropped entirely, do not carry over

`voxels`, `llm`, `model`, `usage`, `reasoning`, `transforms`, `ground_truth`, `export_frame`,
`source`, `files`, `view`, every `bounding_box.*.ratios`, `bounding_box.a_with_b`,
`bounding_box.a_to_b`, and `images.*.camera` / `width` / `height`.

## Gotchas

**`phrasing` is gone.** It recorded which of four shapes the invented-object flow wrote a
phrase in — whether it named both objects, one, or neither. Seeded generation varies `detail`
instead, and always names both, so the field has nothing to hold and is not written. A corpus
built by `build.mjs` still carries one; it is not part of this format and sorts last.

**`textured` is not a rule about roles.** A seeded asset is downloaded whole, so both roles
come back with their own materials and both read `true`. The `anchor: false` of older samples
was a fact about the farm's two lanes — anchors went down the geometry-only one, since only
their shape gets reasoned about — and not something to reproduce.

**`name` flipped meaning.** Old `spec.json` had `name: "server_rack"` (slug) and
`label: "server rack"` (readable). The new file has only `name`, holding the readable
form. Mapping `name` → `name` puts a slug in the wrong field.

**`a` / `b` become `anchor` / `placed`.** `a` is always the anchor and `b` the placed
object — in `files`, `source`, `transforms`, `bounding_box`, `images`, `objects`, and
`SLOTS`.

**Old `files.a` had no extension.** `spec.json` stored bare stems
(`server_rack_f29f9f7d-…`) while the legacy `dataset` metadata stored full names. The new
`image` and `mesh` are always full filenames.

**Images are always PNG.** Anything not already PNG is re-encoded, both as it is made and as
it is collected onto the volume, so a sample's image is always `<stem>.png`. Samples built
before that change keep their original extension, which is why `image` is still a stored
filename rather than something you derive: read the field, do not rebuild it.

**The image is a render of the mesh itself.** `pipeline/render.mjs` draws the seeded asset
from its glTF front (+Z, a 10° downward pitch) against a transparent background, so the photo
and the geometry cannot disagree — they are the same object. Samples from the farm flow carry
Trellis's conditioning frame instead: background removed, square-cropped to the subject and
composited onto black, which is the frame that mesh was actually built from. Expect a black
background and a per-object size on those, not a transparent 512².

**`size` is an array, not `{x, y, z}`.** Legacy `bounding_box.*.dimensions` was an object.
There is no `ratios` field and it should not be recomputed.

**`combined_size` is not derivable from the other two sizes.** It is the union of the two
posed bounding boxes, so it depends on the arrangement. Taking the per-axis max of the two
sizes is wrong whenever the placed object overhangs, and equally wrong the other way: in
the legacy bathtub sample `a_with_b` equalled `a_only` exactly, because the duck sat
entirely inside the tub.

**Sizes are post-bake.** `placed.size` includes the placement scale, so it is not the raw
mesh's bounding box.

**`mesh` names the posed GLB on the volume.** There is no `out/` directory and no
`a.glb` / `b.glb`. The raw mesh stays behind under `datasets/raw/stage1-work/<id>/`, so
baking never overwrites its own input and a sample can be re-placed without re-fetching.

**The placement transform lives inside the GLB, not the JSON.** `bakeTransform`
re-parents the scene under a node named `placement` carrying the TRS, so the numbers are
in `nodes.at(-1)`. Do not add transform fields back to the JSON.

**`metadata.json` is the sentinel that `spec.json` used to be.** A folder counts as a
sample because it has one, and the pipeline's "skip what is already done" checks read it —
`mesh` says the asset landed, `combined_size` says the sample has been placed.

**The pipeline never reads a mesh locally.** `mesh` is a filename, not a path: baking writes
the posed GLB to the volume, and nothing in `pipeline/` expects one on this machine. A sample
folder may still hold its meshes if they were fetched with `--source`, so the viewer asks the
server, which reads the folder first and falls back to `dc-scene-ops`.

**Legacy `dataset/` is not migrated.** Those folders come from the browser exporter and
keep their own shape, with real camera data and hand-placed transforms. Only `generated/`
uses this format, and `run.mjs` refuses any folder without a `metadata.json` — which is
what stops it from baking over a hand-placed arrangement.

**Folders still holding a `spec.json` are invisible to the pipeline.** Samples from before
this format are not migrated and no stage reads them. The viewer does list them, but only to
load the two GLBs sitting in the folder — it reads nothing out of `spec.json`, and those
meshes were never posed, so the pair comes up overlapping at the origin.
