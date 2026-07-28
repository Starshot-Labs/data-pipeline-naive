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
| 4   | Voxelize | an occupancy grid per object                     | seconds, free                   |
| 5   | Place    | where the placed object goes, decided by an LLM  | ~1 min per sample, cents        |
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
| `node pipeline/run.mjs`                     | 4–6    | Voxelizes, places and bakes every unplaced sample.                               |
| `node pipeline/run.mjs <sampleId>`          | 4–6    | Places one specific sample.                                                      |
| `node pipeline/run.mjs --dry`               | 4      | Prints the prompt the LLM would get and stops. Calls nothing, costs nothing.     |
| `node pipeline/upload.mjs`                  | 7      | Publishes everything placed that is not on the volume yet.                       |
| `node pipeline/upload.mjs --force`          | 7      | Re-publishes everything placed.                                                  |

The same commands exist as npm scripts, but note the argument styles differ:
`npm run all -- 50` takes a bare number, `npm run images -- --pairs=50` takes a flag, and
`npm run meshes`, `npm run generate`, `npm run pipeline`, `npm run upload` mirror the rest.

Placement bakes the meshes **in place** and does not keep the unposed originals, so a sample
is placed exactly once — re-running skips anything already posed. Redoing a placement means
rebuilding that sample's meshes. It also refuses any folder without a `metadata.json`, which
is what keeps it away from the hand-placed exports in `dataset/`.

### Looking at results

| Command          | What it does                                       |
| ---------------- | -------------------------------------------------- |
| `npm run dev`    | Runs the API and the web client together.          |
| `npm run server` | API only, port 3000.                               |
| `npm run client` | Web client only, port 5173.                        |
| `npm run build`  | Builds the client into `dist/`.                    |
| `npm start`      | Serves the built client plus the API on port 3000. |

Four pages, all under `http://localhost:5173`:

- `/pipeline.html` — **what you want.** Pick a placed sample to see the anchor and the
  placed object together, each beside its reference image. Nothing there transforms them, so
  what you see is what the files say. Older `spec.json` folders are listed as `(unposed)`:
  their GLBs load, but nothing ever placed them, so they overlap.
- `/viewer.html` — loads any `dataset/` sample's GLBs raw.
- `/place.html` — drop two GLBs in from anywhere on disk, each posed as its own file has it.
  Nothing has to be in `models/` or `dataset/` first, and nothing is sent to the server.
- `/` — the manual tool: load two GLBs, pose them by hand, export a sample.

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
| `VOXEL_RES_ANCHOR` / `VOXEL_RES_PLACED`           | `64` / `32`                                    | Voxel detail. Prompt size grows with the cube of it. |
| `SPEC_MODEL` / `IMAGE_MODEL` / `OPENROUTER_MODEL` | see README                                     | Which model each stage uses.                         |
| `GENERATED_DIR`                                   | `./generated`                                  | Where the local half of each sample gets written.    |
| `SCENE_BASE_URL`                                  | the deployed `dc-scene-ops` endpoint           | Point the pipeline at a different scene service.     |
| `SCENE_REQUEST_TIMEOUT_S`                         | `900`                                          | How long a voxelize or bake call may take.           |

The full list, including every `TRELLIS_*` setting, is in the
[README](README.md#automated-pipeline).
