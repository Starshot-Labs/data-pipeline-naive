# `generated/<id>/metadata.json`

One file per sample, covering the whole pipeline. Replaces `spec.json` and
`out/<id>/result.json`.

```json
{
	"id": "server_rack_f29f9f7d-02ff-4747-b83a-0d42ef995fb8",
	"uuid": "f29f9f7d-02ff-4747-b83a-0d42ef995fb8",
	"created_at": "2026-07-28T00:40:50.607Z",
	"context": "space station module",
	"phrasing": "anchor",
	"placement": "leaning against the bottom right side of the server rack",

	"anchor": {
		"name": "server rack",
		"description": "A tall, rectangular metal server rack painted matte white, featuring modular rack-mount panels on the front and thick horizontal cooling vents along the side panels.",
		"image": "server_rack_f29f9f7d-02ff-4747-b83a-0d42ef995fb8.png",
		"mesh": "server_rack_f29f9f7d-02ff-4747-b83a-0d42ef995fb8.glb",
		"textured": false,
		"size": [0.634936, 1.602104, 0.5512]
	},

	"placed": {
		"name": "power cell",
		"description": "A small, cylindrical battery cell encased in a rugged yellow titanium shell with heavy metallic latches at both flat ends.",
		"image": "power_cell_f29f9f7d-02ff-4747-b83a-0d42ef995fb8.png",
		"mesh": "power_cell_f29f9f7d-02ff-4747-b83a-0d42ef995fb8.glb",
		"textured": true,
		"size": [0.30122, 0.21988, 0.2204]
	},

	"combined_size": [0.81244, 1.602104, 0.5512],

	"intent": {
		"contact": "lean",
		"direction": [0, 0, -1],
		"embed_fraction": 0
	},

	"physics": {
		"contact": "lean",
		"moved": 0.021553,
		"rotated_degrees": 2.4,
		"gap_before": 0.0192,
		"gap_after": -0.0011,
		"flags": []
	}
}
```

`placement.txt` stays alongside it holding the same string as `placement`. Both are
written and must match.

A finished sample lives at `trellis-scene-vol-v2:datasets/raw/stage1/<id>/`, assembled there
by two writers rather than uploaded whole. `dc-scene-ops` writes the posed meshes and their
reference images straight off the farm's volumes; `pipeline/upload.mjs` adds metadata.json
and placement.txt afterwards. Metadata lands **last**, and its presence marks a sample done.

Locally a sample folder holds only metadata.json, placement.txt and the two images. The
meshes never come down — `mesh` names the file as it exists on the volume.

## Fields

- `id` — folder name, `<anchor slug>_<uuid>`
- `uuid` — kept even though `id` ends with it and every filename contains it. Deliberate.
- `created_at` — ISO timestamp from stage 1
- `context` — one of the entries in `CONTEXTS`
- `phrasing` — `both`, `anchor`, `object`, or `bare`
- `placement` — the instruction phrase
- `anchor` / `placed` — one block each, identical shape:
    - `name` — readable name (`"server rack"`), never the slug
    - `description` — one-sentence visual description fed to the image model
    - `image` / `mesh` — filenames including extension
    - `textured` — anchor `false`, placed `true`
    - `size` — `[x, y, z]` of the baked mesh
- `combined_size` — `[x, y, z]` union box of the two posed meshes
- `intent` — the placement model's classification of its own answer (`contact` —
  `rest`, `lean`, `attach`, `embed`, `drape` or `none` — plus `contact_direction` as
  `direction` and `embed_fraction`), recorded so a physics-only re-pass
  (`run.mjs --physics-only`, the viewer's "Apply Physics") knows what the placement
  meant without another model call. A `drape` bakes deformed vertices rather than a new
  transform; the placement TRS in the GLB stays the model's.
- `physics` — what the mesh-contact refinement between place and bake did:
    - `contact` — the placement model's own classification (`rest`, `lean`, `attach`,
      `embed`, `none`), which picked the solver's behaviour
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
      cloth pass refused and the rigid bake of the model's answer stands) mean the pass
      reverted, held something back, or wants review; `error` (with no numbers) means
      it crashed and the model's transforms were baked untouched

Keys appear as stages complete: stage 1 writes everything except `image`, `mesh`,
`textured`, `size` and `combined_size`; stage 2 adds `image`; stage 3 adds `mesh` and
`textured`; stage 6 adds the sizes, `intent` and `physics`. Omit a key that is not ready
rather than writing `null` — absence is how the pipeline knows what work is left.
`physics` is absent when `PLACEMENT_PHYSICS=off` — deliberately deleted then, because a
report left by an earlier pass would describe transforms that are no longer the baked
ones.

**Dropped entirely, do not carry over:** `voxels`, `llm`, `model`, `usage`, `reasoning`,
`transforms`, `ground_truth`, `export_frame`, `source`, `files`, `view`, every
`bounding_box.*.ratios`, `bounding_box.a_with_b`, `bounding_box.a_to_b`, and
`images.*.camera` / `width` / `height`.

## Gotchas

**`name` flipped meaning.** Old `spec.json` had `name: "server_rack"` (slug) and
`label: "server rack"` (readable). The new file has only `name`, holding the readable
form. Mapping `name` → `name` puts a slug in the wrong field.

**`a` / `b` become `anchor` / `placed`.** `a` is always the anchor and `b` the placed
object — in `files`, `source`, `transforms`, `bounding_box`, `images`, `objects`, and
`SLOTS`.

**`phrasing: "object"` still says "object".** The `PHRASINGS` enum was not renamed, so a
sample can read `phrasing: "object"` while its block is called `placed`. Leave it.

**Old `files.a` had no extension.** `spec.json` stored bare stems
(`server_rack_f29f9f7d-…`) while the legacy `dataset` metadata stored full names. The new
`image` and `mesh` are always full filenames.

**Images are always PNG.** Anything not already PNG is re-encoded, both as it leaves the
image model and as it is collected onto the volume, so a sample's image is always
`<stem>.png`. Samples built before that change keep their original extension, which is why
`image` is still a stored filename rather than something you derive: read the field, do not
rebuild it.

**The published image is Trellis's conditioning frame, not the generated photo.** Before
reconstructing, the farm removes the background, square-crops to the subject and composites
onto black — that frame is what the mesh was actually built from, so it is what the dataset
carries. It comes out of the shards in `t2farm-output-v2/<run>/images/`, where each object's
complete PNG sits inside a zstd-compressed safetensors blob. Expect a black background and a
size that varies per object, not the flat white 512×512 the image model produced.

A run whose farm build predates those shards falls back to the photo that was sent up —
same subject, uncropped, original background — and stage 3 prints a warning saying how many
did, since a corpus that silently mixed the two would be worse than either.

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
baking never overwrites its own input and a sample can be re-placed without re-meshing.

**The placement transform lives inside the GLB, not the JSON.** `bakeTransform`
re-parents the scene under a node named `placement` carrying the TRS, so the numbers are
in `nodes.at(-1)`. Do not add transform fields back to the JSON.

**`metadata.json` is the sentinel that `spec.json` used to be.** A folder counts as a
sample because it has one, and the pipeline's "skip what is already done" checks read it —
`mesh` says stage 3 collected that object, `combined_size` says the sample has been placed.

**The pipeline never reads a mesh locally.** `mesh` is a filename, not a path: baking writes
the posed GLB to the volume, and nothing in `pipeline/` expects one on this machine. A sample
folder may still hold its meshes if they were brought down by hand, so the viewer asks the
server, which reads the folder first and falls back to `dc-scene-ops`.

**`view` is no longer persisted.** `images.mjs` draws the azimuth and elevation at render
time instead. Re-rendering an image that failed therefore picks a fresh random angle rather
than reusing the one the first attempt used.

**Legacy `dataset/` is not migrated.** Those folders come from the browser exporter and
keep their own shape, with real camera data and hand-placed transforms. Only `generated/`
uses this format, and `run.mjs` refuses any folder without a `metadata.json` — which is
what stops it from baking over a hand-placed arrangement.

**Folders still holding a `spec.json` are invisible to the pipeline.** Samples from before
this format are not migrated and no stage reads them. The viewer does list them, but only to
load the two GLBs sitting in the folder — it reads nothing out of `spec.json`, and those
meshes were never posed, so the pair comes up overlapping at the origin.
