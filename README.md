# GLB Placement Data Creator

A small client + pipeline for building an ML dataset that describes **how two GLB
models are placed relative to each other**. You load two GLBs, arrange them in 3D
(translate, rotate, scale), write a natural-language placement description, and export a
structured training sample.

For the version that builds samples on its own — invent them, render reference images,
mesh them, then let an LLM decide the placement — see
[Automated pipeline](#automated-pipeline).

**Just want to run it?** [RUNNING.md](RUNNING.md) is the practical guide: every script, what
each one costs, and the end-to-end sequence. This file covers why it is built the way it is.

## What a sample contains

Each export writes a new folder `dataset/<assetA>_<uuid>/`. A single `uuid` is generated
per export and every file is named `<assetName>_<uuid>`:

| File                  | Description                                                                                                                                           |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `<assetA>_<uuid>.glb` | GLB **A** (the _anchor_) with its transform baked in and its bounding-box **center at the origin (0,0,0)**.                                           |
| `<assetB>_<uuid>.glb` | GLB **B** with its transform, expressed **relative to A** (same origin shift applied, so the A↔B arrangement is preserved).                           |
| `<assetA>_<uuid>.png` | Square render of **A only** (B hidden), centered, **front-facing along +Z** with a 10° downward pitch (camera slightly above, so the top is visible). |
| `<assetB>_<uuid>.png` | Square render of **B only** (A hidden), centered, using **your live camera's angle** as the reference direction.                                      |
| `placement.txt`       | The placement string you typed.                                                                                                                       |
| `metadata.json`       | `uuid`, source asset names, placement string, both bounding-box ratios (A-only and A+B), exact transforms, and camera parameters.                     |

### `metadata.json` transforms

- `a_in_export_frame` / `b_in_export_frame` — TRS of A and B after A's bbox center is
  moved to the origin (the same rigid translation is applied to both, matching the GLBs).
- `b_relative_to_a_local` — B's TRS expressed in A's **oriented, bbox-centered** frame.
  This is the clean 6-DoF "B relative to A" target, independent of how the pair was
  posed in world space.
- `bounding_box.a_only` / `bounding_box.b_only` / `bounding_box.a_with_b` — dimensions and
  ratios (normalized so the largest axis = 1) of A alone, B alone, and the A∪B union, all
  post-transform.
- `bounding_box.a_to_b` — per-axis ratio of A's dimensions to B's (how large the anchor is
  relative to B).

To add `b_only` / `a_to_b` to samples exported before this was tracked, run
`npm run backfill` (it recomputes B's box from each sample's exported B `.glb`; set
`DATASET_DIR` to target a different dataset folder).

## Setup

```bash
npm install
```

Drop your `.glb` (or `.gltf`) files into the `models/` folder.

## Run (development)

```bash
npm run dev
```

This starts the Express backend (port 3000) and the Vite dev client (port 5173, with a
proxy to the backend). Open **http://localhost:5173**.

To visually verify that exported A/B GLBs are posed correctly relative to each other,
open the sample viewer at **http://localhost:5173/viewer.html** (also linked from the
creator UI). Pick a sample to load both baked GLBs into one scene.

For a pair that lives nowhere the server can see, **http://localhost:5173/place.html** takes
two GLBs dropped in from anywhere on disk and loads each with the transform baked into its
file, so an already-posed pair stays posed. The gizmo moves them from there, and **Reset**
puts one back to its file pose.

## Run (production build)

```bash
npm run build
npm start            # serves the built client + API on port 3000
```

## Configuration (env vars)

- `MODELS_DIR` — folder scanned for source GLBs (default `./models`).
- `DATASET_DIR` — folder where samples are written (default `./dataset`).
- `PORT` — backend port (default `3000`).

## Usage

1. Pick **A (anchor)** and **B** from the dropdowns.
2. Click an object to select it; drag the gizmo to place it.
    - `W` = move, `E` = rotate, `R` = scale, `F` = frame view, `A`/`B` = select, `Esc` = deselect.
    - Toggle gizmo **Space** (World/Local) and **Snap** as needed.
3. Type the placement description for **B relative to A**.
4. Click **Export sample**.

## Conventions & assumptions

- **Anchor / origin:** A's _bounding-box center_ is placed at the origin on export. A keeps
  whatever rotation you gave it; leave A unrotated (the default) to keep it in canonical pose.
- **Front (+Z):** glTF convention — a model's front faces `+Z`. The A image looks along
  `+Z` toward the origin from **slightly above**, a **10° downward pitch** so the top of the
  object is slightly visible.
- **Images:** both renders are square (`1024²` by default) with the object centered. The A
  image uses the fixed front/pitch view; the B image reuses **your live camera's angle** but
  repositions to center B. Tune `IMAGE_SIZE`, `IMAGE_A_FOV`, and `PITCH_DOWN_DEG` in
  `src/exporter.ts`.
- **Scale:** each GLB's authored scale is preserved on load (no auto-normalization), so
  relative sizes between A and B start faithful. You can then scale either object with the
  gizmo (`R`); the applied scale is baked into the exported GLB and recorded in the metadata
  TRS (`b_in_export_frame.scale` / `b_relative_to_a_local.scale`).
- Exported PNGs have a transparent background and deterministic studio lighting.

## Automated pipeline

`pipeline/` builds samples without anyone posing anything by hand. Seven stages: 1-3 invent
a sample and turn it into two meshes, 4-6 decide how those meshes go together and bake it
in, and 7 publishes the result. Every sample is one self-contained folder described by
[METADATA.md](METADATA.md), which is both what the stages read and what gets uploaded.

One command takes a pair count all the way through:

```bash
node pipeline/all.mjs 50     # or: npm run all -- 50
```

Each stage also runs on its own, which is what you want most of the time — they have
nothing in common operationally:

```bash
node pipeline/build-images.mjs --pairs=50  # 1-2  invent + render   cheap, no GPU, 50 wide
node pipeline/build-meshes.mjs             # 3    mesh              GPU, one campaign at a time
node pipeline/run.mjs                      # 4-6  voxelize, place, bake
node pipeline/upload.mjs                   # 7    publish to the scene volume
```

Splitting 1-2 from 3 matters in practice: images can be prepared at any time, including
while the farm is busy with someone else's campaign, whereas meshing has to wait its turn.
Prepare a corpus first, then start the mesh run when the farm is free. `node
pipeline/build.mjs --pairs=50` still runs 1-3 back to back if you want that.

`OPENROUTER_API_KEY` and `GOOGLE_API_KEY` must both be in `.env` (loaded automatically): the
text stages go through OpenRouter, stage 2 goes straight to Google. A variable already set in
the shell wins over the file, so an inherited `GOOGLE_API_KEY` silently shadows the one in
`.env` — clear it if every image comes back `API key not valid`.

Everything is re-runnable. Each stage skips work already on disk, and `all.mjs` only places
samples that have both meshes and no pose baked in yet, so a second pass costs nothing
beyond filling the gaps a failed image, mesh, placement or upload left behind. `node
pipeline/all.mjs` with no count advances what is already there without inventing more.

### Concurrency and rate limits

Each stage that talks to a model has its own width, defaulting to `200`:
`SPEC_CONCURRENCY` for invented batches, `IMAGE_CONCURRENCY` for renders,
`PLACE_CONCURRENCY` for placements. They are separate because the quotas they answer to are
unrelated — an image quota tightening is no reason to place fewer samples — and setting
`CONCURRENCY` overrides all of them at once when you just want everything throttled.
Publishing is the exception at `50`, since it answers to `dc-scene-ops` rather than a model
and extra width there only deepens a queue.

Only invention actually batches: `SPEC_BATCH` (`20`) pairs go in one structured-output call,
and the total is split across the contexts first, so a 500-pair run comes out as 36 requests
rather than 500. Images and placements are one call per item.

Every item also gets `MODEL_ATTEMPTS` (`3`) goes — one call plus two retries, jittered. That
sits outside the clients' own retry logic, which already handles what the APIs say about
themselves; this layer is for what they call final, like a reply carrying no image or an
answer that fails validation. Those are usually a bad roll, and re-rolling costs one call
rather than the whole extra pipeline pass it otherwise takes to notice.

Meshing does not use it, because there is nothing to widen: every object goes into a single
campaign and generates at the same time. **Campaigns cannot overlap** — the farm's
dispatcher bumps the epoch and clears every queue when one starts, so a second campaign
cancels the first. That is a property of the farm, not a choice here, and it is why the
parallelism lives inside one campaign rather than across several. Before starting, the
stage checks `/status` and, if anything is already running, cancels it with `POST /reset`
rather than waiting it out — a stale campaign would otherwise block forever, and we never
want its results once we have decided to start a new one.

Timeouts are set so the farm's own limits are always what fire first, never a client one:
`TRELLIS_RUN_TIMEOUT_S` is 48 h against a server cap of 24 h and
`TRELLIS_REQUEST_TIMEOUT_S` 10 min for the small calls, while the volume upload and asset
download have no timeout at all. `TRELLIS_PUBLISH_GRACE_S` (1 h) is how long `/report` may
keep answering `404` **before the campaign has ever reported** — `campaign_describe` walks
the entire input Volume before publishing any state, and gets slower as the corpus grows.
Once it has reported, a `404` means something else entirely: the state existed and is gone,
which is the farm going away mid-campaign. That gets `TRELLIS_VANISH_GRACE_S` (2 min),
enough to ride out a redeploy and no more.

**The GLBs, not the report, decide when a campaign is done.** `/report` turning `200` is the
happy path and carries the cost figures, but it cannot be the only trigger: a dispatcher that
dies after writing the meshes, or one that never publishes a final state, would strand a run
that had already produced everything. So the wait also lists `<run_id>/glb` on the output
volume every `TRELLIS_ASSET_POLL_S` (60 s — each check is a CLI subprocess, so it runs slower
than the report poll) and finishes as soon as every object has landed. Because the farm drops
objects whose decoder mesh blows past `FINISH_MAX_FACES`, the count can legitimately stop
short, so a total that has not moved for `TRELLIS_ASSET_SETTLE_S` (3 min) is taken as final.
The same check runs before giving up for any reason, so a campaign that errors or vanishes
still hands back whatever it managed to write.

A campaign is the one piece of work too expensive to lose, so it is checkpointed:
`.trellis-campaign.json` records the run id and the key→mesh mapping as soon as the
campaign starts. If the process dies, or the collect step fails, or you interrupt it, the
next run **attaches to that campaign** instead of starting a second one — no re-upload, no
duplicate GPU spend. The file is removed once the meshes land, and also if the campaign
ends in `error` or `empty`, since re-attaching to those would loop forever.

Nothing is buffered until the end of a stage — each unit is persisted the moment it lands:
a spec folder as its batch returns, an image as it decodes, a `mesh` field as the collect
step reports it, and a placement's sizes before the worker resolves. Being rate limited only
costs the items that were actually in flight.

On top of that, `chat()` in `openrouter.mjs` retries `429` and `5xx` up to
`OPENROUTER_ATTEMPTS` (`5`) times, honouring `Retry-After` and otherwise backing off
exponentially with jitter so fifty workers do not all return at the same instant. Other
4xx responses fail immediately — a bad model id should not be retried.

`nano-banana.mjs` does the same for stage 2, but on three separate budgets, because the ways
a Google call comes back without an image do not mean the same thing. A `429` is the fifty
workers colliding rather than anything about this prompt, so it pushes a **shared deadline**
that every worker waits behind — the one that drew it and the ones about to start — taken
from Google's own `retryDelay`, up to 8 times. An `IMAGE_RECITATION` refusal is a
probabilistic filter that clears on a re-roll, so it gets 15 short capped retries of its own.
Everything genuinely transient — `5xx`, a dropped connection, an empty response whose
`finishReason` is one the model recovers from — spends `GOOGLE_ATTEMPTS` (`5`) with the same
jittered exponential backoff. Keeping the three apart matters: charging a burst of 429s to
the general budget would exhaust it in seconds and leave nothing for a real failure. A
refusal that _is_ a verdict on the prompt (`SAFETY`, `PROHIBITED_CONTENT`) fails at once,
since re-issuing it buys the same verdict.

### 1-3 · From a prompt to two meshes

1. **Invent** (`generate.mjs`) — structured-output calls return batches of samples: an
   anchor object, a smaller object placed against it, a one-sentence visual description of
   each, and the placement phrase. The two objects are always their own fields; what varies
   is how much the phrase repeats them. Each request is assigned one of four phrasings and
   told to use it for every sample it returns, so the set covers each way a real instruction
   might arrive:

    | `phrasing` | Form                            | Example                                                                     |
    | ---------- | ------------------------------- | --------------------------------------------------------------------------- |
    | `both`     | `<object> <placement> <anchor>` | specimen box on top of the industrial centrifuge, near the back left corner |
    | `anchor`   | `<placement> <anchor>`          | on top of the wooden workbench, near the front right corner                 |
    | `object`   | `<object> <placement>`          | wicker basket resting on the top surface, near the left edge                |
    | `bare`     | `<placement>`                   | leaning against the front face, near the right edge                         |

    The two that omit the anchor lean on generic parts — "the top surface", "the left side" —
    rather than naming it. Which one a sample used is recorded as `phrasing`.
    The forms are dealt round-robin rather than drawn per request, because drawing
    independently leaves the mix to luck: a ten-sample run came out eight `anchor` and one
    each of two others. Dealt out, twelve samples give exactly three of each.
    `placement.txt` holds only the phrase, which is exactly what stage 5 consumes and what a
    trained model would be handed at inference. The pair count is split evenly across every
    entry in
    `CONTEXTS` — shuffled, so runs too small to reach them all do not keep landing on the
    same few — and each context's share is chunked into requests of at most `SPEC_BATCH`.
    Every call is told which anchors already exist, duplicate triples are dropped, and the
    resulting shortfall is re-planned, so the set stays varied as it grows into the
    thousands.

2. **Render** (`images.mjs`) — each description goes to nano-banana as a reference photo
   for image-to-3D. Camera framing mirrors the manual exporter: the anchor is shot
   straight-on from the front with a 10° downward pitch, the placed object from a random
   direction (azimuth ±160°, elevation 5-35°). The angle is drawn at render time rather
   than stored, so an image that failed and gets retried simply draws a fresh one.
   The call goes straight to Google's Generative Language API (`nano-banana.mjs`) rather
   than through OpenRouter, because the settings that decide whether an image is any use to
   stage 3 live in `imageConfig` — square framing at 512², the size Trellis reconstructs
   from — and a compatibility layer has nowhere to put them.
3. **Mesh** (`trellis.mjs`) — the batched Trellis 2 farm. Stage the images with `POST
/upload`, start **one** campaign for every object with `POST /run?mode=finish`, poll
   `GET /report` until it turns `200`, then pull the GLBs. Every object in a campaign
   generates simultaneously across the farm's pools, so nothing is sliced up on this side.
   The campaign runs **two lanes**. The farm reads the folder, not a parameter: whatever
   lands in `texture_dir` gets a UV unwrap and PBR bake, whatever lands in `geometry_dir`
   does not. Anchors take the geometry lane, since they only have to be voxelized and
   reasoned against, and placed objects take the texture lane, since those are the ones that
   show. Both go in one campaign because phase 1 is bit-identical between the modes —
   splitting them would pay twice for the input-volume walk, the reservation and the warm.
   Passing `texture_mode` alongside the lane folders is a `400`, so it is never sent.

    The upload is one call per lane, at any size. It deliberately does _not_ go through `POST
/upload`: an HTTP body is capped at 4 GiB and a request at 150 s, which at roughly a
    megabyte an image tops out near 3,500 objects and would force exactly the slicing we do
    not want. Each lane is pushed with `modal volume put`, which has no such ceiling and is
    what the farm's docs prescribe for bulk loads; `/run` then reads those folders off the
    input Volume with no files attached. Staging hard-links rather than copies, so gathering
    100k images costs neither disk nor time, and the `o<n>` numbering runs across both lanes
    rather than restarting per lane — the farm keeps only the file stem in an object id, so
    reused numbers would map two assets onto one job. Assets get their `alphaMode: BLEND`
    stripped on the way in — see `forceOpaqueMaterials` in `glb.mjs`.

    One trap worth knowing: `GET /report` answers `202` while a campaign is still running,
    and `202` is inside `response.ok`, so only `200` means finished.

    Two details worth knowing. **Retrieval needs the Modal CLI**: the farm exposes no HTTP
    route for finished assets, so they come off the `t2farm-output-v2` Volume with `modal
volume get <run_id>/glb`. That means the CLI and workspace credentials have to be
    present, and the pipeline forces `PYTHONIOENCODING=utf-8` on the child because the CLI
    prints a `✓` and exits non-zero when the console codec cannot encode it. **Images are
    uploaded as `o0`, `o1`, …** rather than under their real names: the farm derives each
    object id by slugifying the file stem, so feeding it names that are already slug-safe
    makes the id map straight back to a job instead of having to reverse the slug.

Each sample lands as one self-contained folder:

```
generated/<anchor>_<uuid>/
  metadata.json      everything known about the sample — see METADATA.md
  placement.txt      the phrase on its own, mirroring metadata's `placement`
  <anchor>_<uuid>.png   <anchor>_<uuid>.glb
  <placed>_<uuid>.png   <placed>_<uuid>.glb
```

The two roles are `anchor` and `placed` throughout, and every file in a folder is named
`<role slug>_<uuid>`. The anchor's stem is the folder name by construction.

Knobs: `SPEC_MODEL` (default `google/gemini-3.1-pro-preview`, an OpenRouter id), `IMAGE_MODEL`
(`gemini-3.1-flash-image-preview`, a Google one), `SPEC_BATCH` (`20`),
`SPEC_CONCURRENCY` / `IMAGE_CONCURRENCY` (`200`), `MODEL_ATTEMPTS` (`3`),
`GENERATED_DIR` (`./generated`), plus `TRELLIS_BASE_URL`, `TRELLIS_OUTPUT_VOLUME`
(`t2farm-output-v2`), `TRELLIS_INPUT_VOLUME` (`t2farm-input-v2`), `TRELLIS_MODAL_BIN`
(`modal`), `TRELLIS_STAGE` (`.trellis-stage`),
`TRELLIS_STATE` (`.trellis-campaign.json`), `TRELLIS_POLL_S` (`10`),
`TRELLIS_RUN_TIMEOUT_S` (`172800`), `TRELLIS_REQUEST_TIMEOUT_S` (`600`),
`TRELLIS_PUBLISH_GRACE_S` (`3600`), `TRELLIS_VANISH_GRACE_S` (`120`),
`TRELLIS_ASSET_POLL_S` (`60`) and `TRELLIS_ASSET_SETTLE_S` (`180`). All of them work from
`.env` or the shell.

Budget two ways. Images cost about $0.07 each, so two per sample is ~$140 per thousand
samples. Meshing is billed in GPU-seconds: the farm's own report puts it near 20
GPU-s/object across both phases, and it sustains roughly 1.2 objects/s in phase 1 and 1.0
in phase 2, so a thousand samples is 2000 objects — about half an hour of wall clock rather
than the day the old one-at-a-time endpoint needed.

Expect the farm to drop 5–10% of objects whose decoder mesh blows past `FINISH_MAX_FACES`.
A pair is worth nothing with only one of its meshes, and an oversize drop is deterministic —
the same image yields the same mesh, which will fail the same cap again — so **the sample
folder is deleted** rather than left to be re-uploaded by every later run. That only happens
when the report accounts for every undelivered object with a permanent cause
(`dropped_oversize`, `dropped_degenerate`, `dropped_oom`, `geom_failures`), since the report
counts its drops without naming them. A campaign that also lost something transiently, to a
dead shard or a plain error, deletes nothing and leaves the whole lot for the next run.

### 4-6 · Placement

```bash
npm run pipeline                  # every sample in generated/
npm run pipeline -- <sampleId>    # specific samples
npm run pipeline -- --dry         # print the LLM prompt, call nothing
```

A folder needs a `metadata.json` to be read at all, which is also what keeps this away from
the hand-placed exports in `dataset/`: it bakes in place, and baking over a human's
arrangement would destroy the thing that made it worth keeping.

4. **Voxelize** (`voxelize.mjs`) — triangles are surface-sampled onto a 3×-subdivided
   grid, then a flood fill from the outside marks everything it cannot reach as interior.
   Closed volumes come out solid; open ones (a bathtub basin, the space under a table)
   stay hollow. The sub-grid is collapsed back down and a voxel only counts as occupied
   once more than 20% of its volume is solid, which drops the slivers left where a surface
   just clips a corner. Each grid is centred on its own bounding box.
5. **Place** (`place.mjs`) — both grids are printed as horizontal `#`/`.` slices and sent
   to OpenRouter with the placement phrase. Runs of identical consecutive layers collapse
   into one block headed `y=31..12`, which keeps prismatic shapes (walls, table legs)
   cheap at high resolution. The prompt keeps its own `A`/`B` naming for the pair — `A` is
   the anchor, `B` the placed object — and everything is expressed in **A-voxels**: A's
   grid is the working frame, and the model returns
    - `scale` — how many A-voxels one B-voxel becomes. B's absolute size is deliberately
      undefined, exactly as it will be once both objects come out of Trellis.
    - `yaw_degrees` — rotation about `+Y`.
    - `center` — B's bounding-box centre in A-voxel coordinates.
6. **Bake** (`glb.mjs`, on the scene service) — each GLB's scene is re-parented under a
   single node named `placement` carrying the resolved transform, so both meshes are posed
   purely by their own contents. The anchor is written with its bbox centre at the origin.
   The raw mesh is read from the work prefix and the posed copy written to the published one,
   so baking never overwrites its own input and re-placing a sample costs nothing.

The pose therefore lives in the files rather than in metadata. What lands in `metadata.json`
is the measured outcome: each object's `size` and the `combined_size` of the two together,
which is the only one of the three that cannot be derived from the others, and which is what
marks a sample placed. Sizes are measured from the transformed triangles rather than from a
transformed bounding box, since those only agree when the yaw is a multiple of 90°.

Knobs: `VOXEL_RES_ANCHOR` (default `64`), `VOXEL_RES_PLACED` (`32`), `OPENROUTER_MODEL`
(`google/gemini-3.1-pro-preview`).

Resolution is the cost dial: prompt size grows with the cube of it, and organic shapes
dedupe poorly. At `64`/`32` a wall costs ~700 characters and a lounge chair ~260k, so drop
to `48` or `32` if you are running the whole dataset.

### 7 · Publishing

```bash
npm run upload              # everything placed that is not up yet
npm run upload -- --force   # re-publish everything placed
```

Almost nothing is left to do by this point. Baking already wrote the posed meshes and their
reference images into `trellis-scene-vol-v2:datasets/raw/stage1/<id>/`, so publishing only
adds the two small text files this machine owns — metadata.json and placement.txt. They go
last on purpose: a folder with meshes but no metadata.json is a sample mid-flight, one with
metadata.json is finished, and that is exactly the test the skip check applies.

### The scene service

Everything mesh-shaped happens in `dc-scene-ops`, a separate Modal app:

```bash
npm run deploy      # modal deploy modal/scene_ops.py
```

It mounts the farm's output and input volumes read-only and the scene volume writable, so
collecting a campaign's meshes and images, voxelizing them and baking a pose into them are
all volume-to-volume.

The image a sample publishes is the **frame Trellis actually reconstructed from**, not the
photo the image model produced: background removed, square-cropped to the subject and
composited onto black. That is the one that lines up with the mesh, so it is the one worth
training against.

Those live in `t2farm-output-v2/<run>/images/` as `.st.zst` shards — one zstd frame per
group of objects, wrapping safetensors that hold a `_meta` JSON blob, an `offsets` prefix
sum and every object's complete PNG concatenated into a single `images` array. `collect`
decodes each shard once and writes out the slices verbatim: the shards call themselves
lossless and exact, so re-encoding could only drift from the conditioning the meshes were
built from. Older farm builds do not write them, and collecting from one of those falls back
to the uploaded photo and prints how many objects it did that for. That is worth about 13 MB a sample of round trip: GLBs average 2.9 MB and
never travel, and the reference images are copied out of the farm's input volume rather than
pushed up a second time. What crosses this machine is a voxel grid on the way to the model
and a transform on the way back.

Keeping it out of the farm's app is deliberate. The farm is a shared GPU service whose
campaigns cannot overlap; this is a small, always-idle CPU service that several runs may call
at once. Mixing them would mean redeploying the farm to change a voxel setting, and a crash
in one would take the other's endpoint down with it.

The commands live in `modal/ops.mjs` and run under Node, reusing `pipeline/glb.mjs` and
`pipeline/voxelize.mjs` unchanged — those grids decide what the placement model sees, and a
Python reimplementation could shift them in ways no test would catch. `modal/scene_ops.py` is
the thin wrapper that mounts the volumes, reloads before reads and commits after writes.

Knobs: `SCENE_BASE_URL` (the deployed endpoint), `SCENE_REQUEST_TIMEOUT_S` (`900`), and on
the service side `SCENE_WORK_PREFIX` (`datasets/raw/stage1-work`) and `SCENE_PUBLISH_PREFIX`
(`datasets/raw/stage1`).

### Viewing a run

`npm run dev`, then open **http://localhost:5173/pipeline.html**. Pick a placed sample to
load both posed GLBs into one scene. Nothing in the viewer transforms them — the
arrangement comes entirely from the files, which is the point of looking.

The meshes are read out of the sample's own folder when they are there and off the scene
volume when they are not, and each object's reference image is shown beside its description.
#   d a t a - p i p e l i n e  
 