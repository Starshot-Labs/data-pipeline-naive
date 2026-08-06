# Running the pipeline

A practical guide to every script and how to get from nothing to a finished sample. For
_why_ any of it works the way it does, see [README.md](README.md).

## Setup, once

```bash
npm install
```

Put `OPENROUTER_API_KEY=...` and `GOOGLE_API_KEY=...` in a `.env` file at the repo root — the
text stages go through OpenRouter and the image stage goes straight to Google. The pipeline
scripts read that file, so every knob at the bottom of this page can live there too. The web
server and client do not — `PORT` and `BACKEND_URL` have to be real environment variables.

A variable already set in your shell or user environment beats the file. If every image fails
with `API key not valid`, that is usually a stray `GOOGLE_API_KEY` shadowing the real one:
check with `[Environment]::GetEnvironmentVariable('GOOGLE_API_KEY','User')`.

You also need the **Modal CLI signed in** (`modal setup`) — the mesh stage uploads images
through it. Check with `modal volume list`; if that prints volumes, you are set.

Then deploy the scene service once, and again whenever `pipeline/` or `modal/` changes:

```bash
npm run deploy
```

That is `dc-scene-ops`, which owns everything mesh-shaped. No GLB is ever downloaded here.

## The short version

```bash
node pipeline/all.mjs 50     # invent 50 object pairs and carry them all the way through
npm run dev                  # then open http://localhost:5173/pipeline.html
```

That's the whole thing. Everything below is for when you want to run the halves apart, which
is usually what you want once the GPU stage is involved.

## The seven stages

| #   | Stage    | Makes                                            | Speed and cost                  |
| --- | -------- | ------------------------------------------------ | ------------------------------- |
| 1   | Invent   | two object names, descriptions, placement phrase | seconds, pennies                |
| 2   | Render   | one reference photo per object, via nano-banana  | seconds, ~$0.07 per image       |
| 3   | Mesh     | a GLB per object, via the Trellis farm           | **minutes to hours, GPU money** |
| 4   | Voxelize | a solid-block decomposition per object           | seconds, free                   |
| 5   | Place    | where the placed object goes, decided by an LLM  | ~1 min per sample, cents        |
| 5½  | Refine   | the pose pressed into true mesh contact — rigid-body, or a cloth drape for soft objects | sub-second, free  |
| 6   | Bake     | the placement written into the GLBs              | instant, free                   |
| 7   | Publish  | the finished folders on the scene volume         | seconds, free                   |

Stages 1–2 are cheap and run 50 requests wide. Stage 3 is the expensive one and can only run
**one campaign at a time** across the whole farm. Stages 4–7 are cheap again.

## Every command

### Running the pipeline

| Command                                     | Stages | What it does                                                                     |
| ------------------------------------------- | ------ | -------------------------------------------------------------------------------- |
| `node pipeline/all.mjs 50`                  | 1–7    | Invents 50 pairs and takes them all the way to a published dataset.              |
| `node pipeline/all.mjs`                     | 1–7    | Same, but invents nothing — just pushes what already exists as far as it can go. |
| `node pipeline/build-images.mjs --pairs=50` | 1–2    | Invents 50 pairs and renders their images. No GPU.                               |
| `node pipeline/build-images.mjs`            | 1–2    | Only fills in images that failed earlier.                                        |
| `node pipeline/build-meshes.mjs`            | 3      | Sends every image that has no mesh to the farm. **Spends GPU money.**            |
| `node pipeline/build.mjs --pairs=50`        | 1–3    | Both of the above back to back.                                                  |
| `node pipeline/run.mjs`                     | 4–6    | Voxelizes, places, physics-refines and bakes every unplaced sample.              |
| `node pipeline/run.mjs <sampleId>`          | 4–6    | Places one specific sample.                                                      |
| `node pipeline/run.mjs --dry`               | 4      | Prints the prompt the LLM would get and stops. Calls nothing, costs nothing.     |
| `node pipeline/run.mjs --dry --source=<dir>` | 4     | Same, against a local corpus: folders holding their own meshes voxelize in-process, no scene service involved. |
| `node pipeline/run.mjs --force`             | 4–6    | Re-places samples that are already placed.                                        |
| `node pipeline/run.mjs --physics-only`      | 5½–6   | Re-runs just the physics on already-placed samples, continuing from their baked poses. No model call. |
| `node pipeline/upload.mjs`                  | 7      | Publishes everything placed that is not on the volume yet.                       |
| `node pipeline/upload.mjs --force`          | 7      | Re-publishes everything placed.                                                  |

The same commands exist as npm scripts, but note the argument styles differ:
`npm run all -- 50` takes a bare number, `npm run images -- --pairs=50` takes a flag, and
`npm run meshes`, `npm run generate`, `npm run pipeline`, `npm run upload` mirror the rest.

### Seeded scene generation (the Objaverse flow)

The retrieval pipeline replaces stages 1–3: instead of inventing objects, rendering them and
meshing them on the farm, scenes are seeded with real Objaverse assets whose uid — and exact
GLB path on Hugging Face — is recorded in the sample at birth. Two scripts, both resumable:

| Command                                              | What it does                                                                    |
| ---------------------------------------------------- | ------------------------------------------------------------------------------- |
| `node pipeline/objaverse-pool.mjs`                   | One-time: builds `pool/pool.jsonl` from TRELLIS-500K's curated 168k Sketchfab subset (captions + aesthetic ≥ 5.5), joins licenses off Objaverse's metadata shards, and LLM-tags every asset with the placement categories it can serve. |
| `node pipeline/objaverse-pool.mjs --limit=800 --tag-limit=240 --no-licenses` | The cheap smoke version of the above.                   |
| `node pipeline/generate-scenes.mjs --samples=50000`  | Generates scenes dealt across the six placement categories by weight — rigid ×2, soft ×1.5, the rest ×1 — with **both** objects drawn from pool assets, uid and GLB path recorded at birth. Re-running fills toward the same quotas. |
| `node pipeline/generate-scenes.mjs --samples=12 --dry` | Prints one composed prompt per category and phrase style, calls nothing.       |
| `node pipeline/fetch-assets.mjs`                     | Downloads every sample's recorded Objaverse GLBs from Hugging Face onto the scene volume and renders each one's reference image, cached per uid. Roles without a uid (corpora from before seeding was mandatory) are counted and skipped. |
| `node pipeline/fetch-assets.mjs --source=<dir> [id ...]` | Same, into a local corpus's own folders — cache lands in `.objaverse-cache/`. |
| `modal run --detach modal/pipeline.py --samples 50000` | The whole retrieval pipeline on Modal, end to end: pool → scenes → fetch + render → place → publish, with everything on the scene volume. |

Diversity is dealt, not hoped for: category quotas are exact and weighted, contexts are
shuffled and dealt within each category, and each category's relations rotate round-robin
across its samples. The code deals only abstract knobs — a pure relation primitive ("on top
of", "inside") and a **detail level**: bare (names and relation only), position (the model
adds one plain direction of its choosing), or part (the model ties the placement to a real
geometric part of the chosen asset, read off its caption — "the middle shelf", "behind the
spare tire"). The simple half of each category deals bare/position, the complex half
position/part; which side, which part, how deep and what orientation is always the model's
per-pairing decision. Every phrase names both objects briefly in plain words, which is
validated in code along with seed uids, articles, neutral verbs, colour-free names and
phrase length; anything dropped is re-planned, up to six rounds. The samples carry
`category`, `complexity`, `detail`, `relation` and per-role `objaverse` blocks in their
metadata; the fetch stage turns those uids into meshes and images.

The reference images come from `pipeline/render.mjs`, a dependency-free software renderer:
each asset is drawn textured from its glTF front (+Z, the exporter's 10° downward pitch)
against a transparent background, lit by one directional sun with a shadow map and a soft
catcher disc on the ground plane. No Blender, no GPU — it runs identically on a laptop and
inside a Modal CPU container, at well under a second per asset.

Every placement — from the CLI or the viewer's button alike — also drops a timestamped
Markdown log into `placement-logs/`: the exact system and user prompt the model saw, the
full answer it gave, what the physics pass did to it, and the transforms that were baked.
Re-placements accumulate rather than overwrite, so the folder is the history of every
decision ever taken.

The physics pass sits between the answer and the bake. The model reasons over voxel
blocks a sixteenth of the anchor wide, so its poses are right in intent and off in
contact — floating a sliver above a surface, or sunk a sliver into it. The pass rebuilds
contact from the actual triangles: the model classifies its own placement (`contact` in
the answer — rest, lean, attach, embed or none) and `pipeline/physics.mjs` drops, seats,
presses or buries the placed object accordingly, touching position and rotation only,
never scale, and never the anchor. The instruction outranks the simulation: a rest drops
straight down first, and the dynamics that follow may only stabilise the pose locally —
on a domed or sloped surface where honest physics would slide the object away from where
the phrase put it, the slide is undone and flagged (`settled_away`) rather than baked.
It is deliberately conservative in general — a solve that finds no contact, wanders too
far, or starts deep inside the anchor reverts to the model's transforms and flags the
sample (`metadata.physics.flags`), so a batch can be filtered instead of silently
damaged. `PLACEMENT_PHYSICS=off` skips the pass entirely.

Soft objects take a different pass: a `drape` contact (a blanket over a bed, a towel on
a rod) runs `pipeline/cloth.mjs` instead — the placed object's own voxelization becomes
a lattice of particles joined by stretch, shear and bend constraints, falls against the
anchor under gravity with friction, and the render mesh follows by trilinear embedding.
The deformed vertices are written back into the GLB in its own local frame under the
unchanged placement TRS, so a draped file behaves exactly like a rigidly-posed one to
everything downstream — textures, the viewer, pose read-back and re-passes included. A
drape that fails falls back to the rigid bake with `drape_failed` in the flags.

The model's classification can be overridden per run: the viewer's **Contact** picker (or
`PLACEMENT_CONTACT` from the CLI) forces any contact onto the next placement or physics
pass — forcing `drape` is how an object the model called a rest gets a soft-body pass,
and the override is recorded as the sample's intent so later re-passes keep it.

Placement skips anything already posed; `--force` re-places it, and baking never consumes
its input either way. A volume-resident sample reads the raw mesh from the work prefix and
writes the posed copy to the published one; a **local corpus** writes each posed mesh as a
`.posed.glb` sibling next to the untouched raw one, so re-placing is always repeatable
(what it does overwrite is the previous posed copy, and the measured sizes in
`metadata.json`). Placement also refuses any folder without a `metadata.json`, which is
what keeps it away from the hand-placed exports in `dataset/`.

### Looking at results

| Command          | What it does                                       |
| ---------------- | -------------------------------------------------- |
| `npm run dev`    | Runs the API and the web client together.          |
| `npm run server` | API only, port 3000.                               |
| `npm run client` | Web client only, port 5173.                        |
| `npm run build`  | Builds the client into `dist/`.                    |
| `npm start`      | Serves the built client plus the API on port 3000. |

Four pages, all under `http://localhost:5173`:

- `/pipeline.html` — **what you want.** Pick a **corpus** (any repo folder starting with
  `generated`; the env `GENERATED_DIR` stays the default) and a placed sample to see the
  anchor and the placed object together, each beside its reference image. Nothing there
  transforms them, so what you see is what the files say. **Compare ⇆** opens
  `/compare.html`: two of these viewers side by side, each with its own corpus and sample
  pickers, for judging one run against another — swap panes from the top bar. Three placing buttons, each a slice of stages 4–6:
  **Place with LLM** asks the model alone with physics forced off (`OPENROUTER_API_KEY`
  must be in `.env`); **Apply Physics** presses the pose currently on screen into mesh
  contact — no model call, using the contact intent recorded when the sample was placed;
  **Full Place** does both in one go. On a local corpus every one of them writes
  `.posed.glb` copies beside the raw meshes, so they can be clicked as often as you like —
  place once, then re-apply physics repeatedly, or re-roll the whole thing. After Full
  Place or Apply Physics, a translucent **Ghost** of the placed object marks the pose the
  physics started from — the model's raw answer — toggleable from the Show row. The **Model** picker — OpenRouter's catalog, filtered to models that support
  structured outputs, searchable via the filter box — decides which LLM answers that click,
  overriding `OPENROUTER_MODEL` for it. **Reset** shows the pair as it was before any
  placement (raw meshes, parked apart), and placing does the same reset first so the reload
  afterwards shows exactly what changed. **Voxel resolution** applies to both the overlays
  and the next placement (it becomes `VOXEL_RES_*` for that click — higher costs more
  prompt). **Reasoning** turns the model's thinking off or sets its effort for that click
  (`OPENROUTER_REASONING`; "model default" sends nothing). The selected run, model,
  resolution and reasoning live in the URL, so a reload or a shared link lands on the same
  view. **Blocks** overlays the exact solid-block decomposition the placement model was
  shown, computed on demand from the sample's own meshes at the pipeline's resolutions, and
  it follows each mesh's baked pose; **Voxels** shows the raw occupancy grid those blocks
  were merged from, one cube per solid voxel. A pair whose loaded files carry no baked pose — an
  `(unposed)` folder, or a local sample not yet re-placed — is parked side by side rather
  than left overlapping at the origin.
- `/viewer.html` — loads any `dataset/` sample's GLBs raw.
- `/place.html` — drop two GLBs in from anywhere on disk, each posed as its own file has it.
  Nothing has to be in `models/` or `dataset/` first, and nothing is sent to the server.
- `/` — the manual tool: load two GLBs, pose them by hand, export a sample.

### Testing placement end-to-end on a local corpus

A sample folder that holds its own meshes needs no volume and no scene service — voxelize
and bake both run in-process against the files, and the bake writes `.posed.glb` copies
beside the raw meshes rather than touching them. Point a stack at the corpus and place
from the page:

```powershell
# a second stack avoids a port clash if `npm run dev` is already up
$env:GENERATED_DIR='<absolute path to the corpus>'
$env:PORT='3100'; $env:BACKEND_URL='http://localhost:3100'
npm run dev        # → the printed client URL, /pipeline.html
```

Pick a sample, click **Place with LLM**, and the viewer reloads with the new pose — the
meshes are served straight out of the sample folder, posed copies preferred. The same
thing from the terminal is `node pipeline/run.mjs --force --source=<corpus> <sampleId>`.
Re-placing overwrites only the posed copies and the measured sizes in `metadata.json`;
the raw meshes and everything stage 1-3 wrote stay as they were.

### Odds and ends

| Command            | What it does                                                |
| ------------------ | ----------------------------------------------------------- |
| `npm run backfill` | Adds missing bounding-box fields to old `dataset/` samples. |

## Doing it in two halves

This is the normal way to work, because the cheap half and the GPU half want different
timing — you can prepare images whenever, but meshing has to wait for the farm to be free.

```bash
# 1. Prepare as many samples as you like, whenever you like. No GPU.
node pipeline/build-images.mjs --pairs=200

# 2. When the farm is free, mesh them. One campaign, all objects at once.
node pipeline/build-meshes.mjs

# 3. Place them.
node pipeline/run.mjs

# 4. Publish them.
node pipeline/upload.mjs

# 5. Look at them.
npm run dev        # → http://localhost:5173/pipeline.html
```

Each step prints what to run next when it finishes.

## Re-running is always safe

Every stage skips work that is already on disk, so re-running only fills gaps:

- a sample folder without images → gets images
- an image with no `mesh` recorded → gets meshed
- a sample with meshes and no `combined_size` → gets placed
- a placed sample with no metadata.json on the volume → gets published

So if a run dies halfway, or rate limits eat some of it, just run the same command again.
Nothing is redone and nothing is lost.

The one thing that _is_ thrown away is a pair the farm refuses to mesh. An oversize drop
happens again every time the same image is submitted, and half a pair is useless, so the
mesh stage deletes the sample folder and prints `– discarded <id>`. It only does that when
the report blames every missing object on a permanent cause; anything that merely failed
gets left alone and retried.

The mesh stage goes further: it writes `.trellis-campaign.json` the moment a campaign starts,
so if the process dies while the farm is still working, the next run **attaches to that same
campaign** instead of starting a second one. You never pay twice for the same objects.

## Where everything lands

```
generated/<anchor>_<uuid>/     the small half of a sample, kept locally
  metadata.json                  everything known about it — see METADATA.md
  placement.txt                  the instruction, mirroring metadata's `placement`
  <anchor>_<uuid>.png            the anchor's reference image
  <placed>_<uuid>.png            the placed object's reference image

dataset/                       hand-made samples from the manual tool
models/                        source GLBs the manual tool loads
```

The meshes are not here. They live on the scene volume, written straight from the farm's
volumes by `dc-scene-ops`:

```
trellis-scene-vol-v2:datasets/raw/stage1-work/<id>/    raw meshes and images, as collected
trellis-scene-vol-v2:datasets/raw/stage1/<id>/         the finished sample
```

`.trellis-campaign.json` and `.trellis-stage/` are working files for the mesh stage. Both
are gitignored, and both clean themselves up.

## When something goes wrong

**"cancelling campaign …"** — something was already on the farm when the mesh stage started,
so it `/reset` that run and continues with yours. Expected whenever you re-run after a
stuck or abandoned campaign.

**You cancelled the mesh stage but the money is still going.** Killing the local process does
not stop the campaign — that lives on Modal. `POST /reset` on the farm cancels it. If you let
it finish instead, the next `build-meshes.mjs` will collect the results you already paid for.

**The mesh stage sits on one progress line and never moves.** The line reprints when the
state or the object count changes, so a campaign still working through its first objects has
genuinely nothing new to say. It is polling either way, and it now also watches the output
volume, so it will collect the meshes even if the farm never reports itself finished. To see
for yourself where a run is:

```powershell
$base = 'https://starshot-aitools--trellis2-latent-farm-web.modal.run'
Invoke-WebRequest "$base/report?run_id=<run id>" -UseBasicParsing | Select-Object StatusCode
modal volume ls t2farm-output-v2 <run id> --env main
```

`202` means it is genuinely still working. `modal-http: invalid function call` on every
route means the farm app is not deployed — check `modal app list --env main` for a
`trellis2-latent-farm` in state `deployed`, and if there is none, the campaign is gone and
someone has to redeploy the farm. A run id whose folder on the output volume has `latents`
and `occupancy` but no `glb` never reached phase 2, so there is nothing to download.

**Port 3000 is taken.** Run the API somewhere else and point the client at it:

```powershell
$env:PORT=3100; $env:BACKEND_URL='http://localhost:3100'; npm run dev
```

**Meshes look untextured.** Anchors are meant to be — they go down the geometry-only lane,
since only their shape matters. Placed objects get the texture bake. If a _placed_ object
comes back untextured, check `phase2.shards_by_mode` in the farm's report.

**Rate limits.** OpenRouter calls back off and retry on their own. So do the image calls,
except that a `429` from Google parks every worker behind one shared deadline instead of just
the one that hit it — so the render stage going quiet for half a minute is it waiting out a
quota, not hanging.

On top of that, every invented batch, image, placement and publish gets `MODEL_ATTEMPTS`
(`3`) goes in total. That outer retry catches what the clients treat as final — a reply with
no image, one that will not parse, an answer that fails validation — since those are usually
a bad roll rather than a bad request. Anything that exhausts its attempts is logged per item
and picked up by the next run, as before.

## Knobs worth knowing

| Variable                                          | Default                                        | Why you'd change it                                  |
| ------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------- |
| `SPEC_CONCURRENCY` / `IMAGE_CONCURRENCY` / `PLACE_CONCURRENCY` | `200` each                        | How wide each model stage runs.                      |
| `PUBLISH_CONCURRENCY`                             | `50`                                           | Publishing answers to scene-ops, not a model quota.  |
| `CONCURRENCY`                                     | unset                                          | Overrides all four at once — one lever to throttle.  |
| `MODEL_ATTEMPTS`                                  | `3`                                            | One call plus two retries, per item.                 |
| `VOXEL_RES_ANCHOR` / `VOXEL_RES_PLACED`           | `16` / `16`                                    | Voxel detail. Prompt cost follows the block count it produces. |
| `PLACEMENT_PHYSICS`                               | `on`                                           | `off` bakes the model's transforms untouched — the A/B switch for the refine pass. |
| `PHYSICS_SDF_RES`                                 | `128`                                          | Cells along the anchor's longest side in the refine pass's distance field. |
| `PHYSICS_SAMPLES`                                 | `2000`                                         | Surface samples on the placed object; contact fidelity and solve cost both follow it. |
| `PHYSICS_WORKERS`                                 | cores − 1, at most 16                          | Worker threads for in-process refines (local corpora and the Modal pipeline). |
| `DRAPE_RES`                                       | `24`                                           | Lattice cells along the placed object's longest side in the cloth pass.       |
| `PLACEMENT_CONTACT`                               | unset                                          | Forces the physics contact (`rest`, `lean`, `attach`, `embed`, `drape`, `none`), overriding the model's classification — the viewer's Contact picker sets it per click. |
| `SPEC_MODEL` / `IMAGE_MODEL` / `OPENROUTER_MODEL` | see README                                     | Which model each stage uses.                         |
| `GENERATED_DIR`                                   | `./generated`                                  | Where the local half of each sample gets written.    |
| `PLACEMENT_LOG_DIR`                               | `./placement-logs`                             | Where every placement's full prompt and answer land. |
| `OPENROUTER_REASONING`                            | `off`                                          | Placement thinking: disabled by default; `low`, `medium` or `high` turn it on, `default` restores the provider's own behaviour. |
| `SCENE_BASE_URL`                                  | the deployed `dc-scene-ops` endpoint           | Point the pipeline at a different scene service.     |
| `SCENE_REQUEST_TIMEOUT_S`                         | `900`                                          | How long a voxelize or bake call may take.           |
| `POOL_DIR`                                        | `./pool`                                       | Where the Objaverse seed pool and its caches live.   |
| `POOL_MODEL`                                      | `openai/gpt-5.6-luna`                          | The pool tagger. Chosen because thinking can be disabled on it — Gemini's flash tier refuses, and mandatory reasoning made tagging ~20× dearer. |
| `POOL_REASONING`                                  | `off`                                          | Tagger thinking: disabled by default; `low`/`medium`/`high` or `default` to restore. |
| `POOL_BATCH` / `POOL_CONCURRENCY`                 | `40` / `200`                                   | Captions per tagging call, and how wide tagging runs. |
| `POOL_LICENSES`                                   | unset (allow all)                              | Exact-match license allow-list for seeds, e.g. `by,by-sa,cc0`. |
| `SPEC_REASONING`                                  | `low`                                          | Scene-generation thinking effort. `off` only works on models that permit disabling — `gemini-3.6-flash` does not. |
| `FETCH_CONCURRENCY`                               | `12`                                           | Samples fetching at once; each is a few MB off Hugging Face. |
| `RENDER_SIZE`                                     | `512`                                          | Reference image edge; rendered at 2× and downsampled. |
| `SCENE_CACHE_PREFIX` / `FETCH_CACHE`              | `datasets/raw/objaverse-cache` / `.objaverse-cache` | Where fetched GLBs and renders cache, on the volume / beside a local corpus. |

The full list, including every `TRELLIS_*` setting, is in the
[README](README.md#automated-pipeline).
