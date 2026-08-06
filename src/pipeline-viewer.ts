import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { addStudioLighting } from './viewer';

type Vec3 = [number, number, number];
type Layer = 'anchor' | 'placed';

interface Part {
  name: string;
  description: string;
  image?: string;
  mesh: string;
  textured: boolean;
  size: Vec3;
}

/** A placed sample's metadata.json — see METADATA.md. */
interface Sample {
  id: string;
  context: string;
  phrasing: string;
  placement: string;
  anchor: Part;
  placed: Part;
  combined_size: Vec3;
}

/** One entry from `/api/runs`: a folder in generated/ and the pair of GLBs it offers. */
interface Run {
  id: string;
  meshes: string[];
  /** False for a folder still on the old spec.json format, whose meshes are unposed. */
  placed: boolean;
}

/** One object's voxel geometry from `/api/blocks`, in its bbox-centred frame. */
interface Grid {
  voxelSize: number;
  origin: Vec3;
  center: Vec3;
  blocks: [number, number, number, number, number, number][];
  voxels: [number, number, number][];
}

/** Which rendering of the voxel geometry is overlaid, if any. */
type OverlayMode = 'blocks' | 'voxels';

const LAYERS: Layer[] = ['anchor', 'placed'];
const TAGS: Record<Layer, string> = { anchor: 'tag-a', placed: 'tag-b' };

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node as T;
}

const dom = {
  viewport: el<HTMLDivElement>('viewport'),
  corpusSelect: el<HTMLSelectElement>('corpusSelect'),
  splitView: el<HTMLButtonElement>('splitView'),
  runSelect: el<HTMLSelectElement>('runSelect'),
  reloadRuns: el<HTMLButtonElement>('reloadRuns'),
  runHint: el<HTMLParagraphElement>('runHint'),
  modelSearch: el<HTMLInputElement>('modelSearch'),
  modelSelect: el<HTMLSelectElement>('modelSelect'),
  resInput: el<HTMLInputElement>('resInput'),
  reasoningSelect: el<HTMLSelectElement>('reasoningSelect'),
  contactSelect: el<HTMLSelectElement>('contactSelect'),
  placeRun: el<HTMLButtonElement>('placeRun'),
  physicsRun: el<HTMLButtonElement>('physicsRun'),
  fullRun: el<HTMLButtonElement>('fullRun'),
  resetRun: el<HTMLButtonElement>('resetRun'),
  placeLog: el<HTMLParagraphElement>('placeLog'),
  frameView: el<HTMLButtonElement>('frameView'),
  toggleBlocks: el<HTMLButtonElement>('toggleBlocks'),
  toggleVoxels: el<HTMLButtonElement>('toggleVoxels'),
  toggleGhost: el<HTMLButtonElement>('toggleGhost'),
  blockOpacity: el<HTMLInputElement>('blockOpacity'),
  status: el<HTMLParagraphElement>('status'),
  selInfo: el<HTMLDivElement>('selInfo'),
  placement: el<HTMLParagraphElement>('placement'),
  numbers: el<HTMLDivElement>('numbers'),
  descriptions: el<HTMLDivElement>('descriptions'),
  llmSelect: el<HTMLSelectElement>('llmSelect'),
  llmHint: el<HTMLParagraphElement>('llmHint'),
  llmLog: el<HTMLDivElement>('llmLog'),
};

const buttons: Record<Layer, HTMLButtonElement> = {
  anchor: el<HTMLButtonElement>('toggleAnchor'),
  placed: el<HTMLButtonElement>('togglePlaced'),
};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
dom.viewport.appendChild(renderer.domElement);

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
addStudioLighting(scene);

// Only the block overlays opt into shadows, so making the key light cast them changes
// nothing for the plain meshes — it is what lets an opaque decomposition read as structure.
const keyLight = scene.getObjectByName('__key') as THREE.DirectionalLight;
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.camera.near = 0.5;
keyLight.shadow.camera.far = 20;
Object.assign(keyLight.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3 });
keyLight.shadow.camera.updateProjectionMatrix();
keyLight.shadow.bias = -0.0005;

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 5000);
camera.position.set(2.5, 1.8, 3.2);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;

const loader = new GLTFLoader();

const layers: Partial<Record<Layer, THREE.Object3D>> = {};
const visible: Record<Layer, boolean> = { anchor: true, placed: true };
let runs: Run[] = [];

/** Which corpus — which `generated*` folder — the viewer reads. '' is the server default. */
let corpus = '';

/** The corpus as a query-string tail for the routes that take one. */
const corpusQuery = () => (corpus ? `&corpus=${encodeURIComponent(corpus)}` : '');

/** Where a sample's plain files (metadata, images) are served from. */
const sampleBase = (id: string) =>
  corpus ? `/file/${encodeURIComponent(corpus)}/${encodeURIComponent(id)}` : `/generated/${encodeURIComponent(id)}`;

const overlays: Partial<Record<Layer, THREE.Group>> = {};
const gridsCache = new Map<string, Record<Layer, Grid>>();
let overlayMode: OverlayMode | null = null;

/** When set, meshes load as their raw files — the pair before any placement. */
let showRaw = false;

/** The placed object's transform from before a physics pass, as `/api/place` returns it.
 *  `raw` marks a drape's ghost: the pose is unchanged there and the geometry is what
 *  deformed, so the ghost is the raw, undeformed mesh at that same pose. */
interface GhostTRS {
  position: number[];
  rotation: number[];
  scale: number[];
  raw?: boolean;
}

// The ghost: a translucent copy of the placed object left at the pose the physics pass
// started from, so a refinement can be read against the answer it corrected. It exists
// per placement click and dies with the next click, reset or sample switch.
let ghostObject: THREE.Object3D | null = null;
let ghostVisible = true;
/** A raw-mesh ghost owns its geometries; a clone shares them with the loaded layers. */
let ghostOwnsGeometry = false;

const ghostMaterial = new THREE.MeshStandardMaterial({
  color: 0x5ec8ff,
  transparent: true,
  opacity: 0.35,
  depthWrite: false,
  roughness: 0.6,
  metalness: 0,
  // Pulled a hair toward the camera so a ghost nearly coincident with the real mesh
  // reads as a tint instead of z-fighting shimmer.
  polygonOffset: true,
  polygonOffsetFactor: -1,
});

function clearGhost(): void {
  if (ghostObject) {
    ghostObject.removeFromParent();
    if (ghostOwnsGeometry) {
      ghostObject.traverse((node) => {
        const mesh = node as THREE.Mesh;
        if (mesh.geometry) mesh.geometry.dispose();
      });
    }
    ghostObject = null;
  }
  ghostOwnsGeometry = false;
  dom.toggleGhost.disabled = true;
  dom.toggleGhost.classList.remove('active');
}

function ghostify(object: THREE.Object3D, ownsGeometry: boolean): void {
  object.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (mesh.isMesh) {
      mesh.material = ghostMaterial;
      mesh.castShadow = false;
      mesh.receiveShadow = false;
    }
  });
  ghostObject = object;
  ghostOwnsGeometry = ownsGeometry;
  ghostVisible = true;
  object.visible = true;
  scene.add(object);
  dom.toggleGhost.disabled = false;
  dom.toggleGhost.classList.add('active');
}

/**
 * Builds the ghost. A rigid pass's ghost clones the freshly loaded placed mesh — same
 * geometry, one shared translucent material — and rewinds its `placement` node to the
 * pre-physics transform; cloning the current load matters because switching samples
 * disposes geometries, and the ghost must share the lifetime of the meshes it shadows.
 * A drape's ghost (`raw`) instead fetches the raw, undeformed mesh and poses it at the
 * same transform, so the overlay shows what the cloth pass did to the shape.
 */
async function showGhost(trs: GhostTRS): Promise<void> {
  clearGhost();

  if (trs.raw) {
    const id = dom.runSelect.value;
    const mesh = runs.find((run) => run.id === id)?.meshes[1];
    if (!mesh) return;
    const loaded = await loader.loadAsync(
      `/mesh/${encodeURIComponent(id)}/${encodeURIComponent(mesh)}?raw=1&v=${Date.now()}${corpusQuery()}`,
    );
    // Raw files carry no placement node — the pose goes straight onto the root.
    loaded.scene.position.fromArray(trs.position);
    loaded.scene.quaternion.fromArray(trs.rotation);
    loaded.scene.scale.fromArray(trs.scale);
    ghostify(loaded.scene, true);
    return;
  }

  const source = layers.placed;
  if (!source) return;
  const clone = source.clone(true);
  const placement = clone.getObjectByName('placement') ?? clone;
  placement.position.fromArray(trs.position);
  placement.quaternion.fromArray(trs.rotation);
  placement.scale.fromArray(trs.scale);
  ghostify(clone, false);
}

function resize(): void {
  const { clientWidth: w, clientHeight: h } = dom.viewport;
  if (!w || !h) return;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
new ResizeObserver(resize).observe(dom.viewport);
resize();

(function animate(): void {
  requestAnimationFrame(animate);
  orbit.update();
  renderer.render(scene, camera);
})();

function setStatus(text: string, kind: '' | 'ok' | 'err' | 'busy' = ''): void {
  dom.status.textContent = text;
  dom.status.className = `status ${kind}`.trim();
}

function clearOverlays(): void {
  for (const key of LAYERS) {
    const overlay = overlays[key];
    if (!overlay) continue;
    overlay.removeFromParent();
    overlay.traverse((node) => {
      const mesh = node as THREE.Mesh;
      if (mesh.geometry) mesh.geometry.dispose();
    });
    delete overlays[key];
  }
}

function clearLayers(): void {
  clearOverlays();
  for (const object of Object.values(layers)) {
    scene.remove(object);
    object.traverse((node) => {
      const mesh = node as THREE.Mesh;
      if (mesh.geometry) mesh.geometry.dispose();
    });
  }
  for (const key of LAYERS) delete layers[key];
}

function applyVisibility(): void {
  for (const key of LAYERS) {
    if (layers[key]) layers[key]!.visible = visible[key];
    buttons[key].classList.toggle('active', visible[key]);
    buttons[key].disabled = !layers[key];
  }
}

function frame(): void {
  const box = new THREE.Box3();
  for (const key of LAYERS) {
    if (visible[key] && layers[key]) box.expandByObject(layers[key]!);
  }
  if (box.isEmpty()) return;
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const direction = new THREE.Vector3(0.6, 0.45, 1).normalize();
  const distance = (sphere.radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.45;
  orbit.target.copy(sphere.center);
  camera.position.copy(sphere.center).addScaledVector(direction, Math.max(distance, 0.4));
  camera.near = Math.max(distance / 100, 0.001);
  camera.far = distance * 100;
  camera.updateProjectionMatrix();
  orbit.update();
}

// One material set per layer, shared by every overlay ever built, so the opacity slider
// adjusts the cubes already on screen and whatever gets built later alike. Standard rather
// than basic material: the cubes shade under the studio lights and shadow one another,
// which is what keeps the structure legible when the fill goes opaque.
const overlayMaterials: Record<Layer, { fill: THREE.MeshStandardMaterial; line: THREE.LineBasicMaterial }> = {
  anchor: {
    fill: new THREE.MeshStandardMaterial({
      color: 0xff8f5e, transparent: true, opacity: 0.18, depthWrite: false, roughness: 0.55, metalness: 0,
    }),
    line: new THREE.LineBasicMaterial({ color: 0xff8f5e, transparent: true, opacity: 0.8 }),
  },
  placed: {
    fill: new THREE.MeshStandardMaterial({
      color: 0x5ec8ff, transparent: true, opacity: 0.18, depthWrite: false, roughness: 0.55, metalness: 0,
    }),
    line: new THREE.LineBasicMaterial({ color: 0x5ec8ff, transparent: true, opacity: 0.8 }),
  },
};

/**
 * One object's merged blocks as shaded cubes with hard edges. The coordinates arrive in the
 * object's bbox-centred frame, so adding the centre back puts them in the raw file's own
 * frame — the same frame the GLB's nodes speak.
 */
function buildBlocksOverlay(layer: Layer, grid: Grid): THREE.Group {
  const group = new THREE.Group();
  group.name = `${layer}-blocks`;
  const { fill, line } = overlayMaterials[layer];

  // A constant inset per face (a few percent of one voxel) leaves the same hairline seam
  // between neighbouring blocks whatever their sizes, so the decomposition reads as pieces.
  const inset = grid.voxelSize * 0.03;
  const [cx, cy, cz] = grid.center;
  for (const [x0, y0, z0, x1, y1, z1] of grid.blocks) {
    const box = new THREE.BoxGeometry(x1 - x0 - 2 * inset, y1 - y0 - 2 * inset, z1 - z0 - 2 * inset);
    const mesh = new THREE.Mesh(box, fill);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.position.set(cx + (x0 + x1) / 2, cy + (y0 + y1) / 2, cz + (z0 + z1) / 2);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(box), line);
    edges.position.copy(mesh.position);
    group.add(mesh, edges);
  }
  return group;
}

/**
 * The raw occupancy: one cube per solid voxel, instanced so thousands cost one draw call.
 * Each cube is shrunk a touch so the lattice reads without needing edge lines.
 */
function buildVoxelsOverlay(layer: Layer, grid: Grid): THREE.Group {
  const group = new THREE.Group();
  group.name = `${layer}-voxels`;
  const side = grid.voxelSize * 0.94;
  const cubes = new THREE.InstancedMesh(
    new THREE.BoxGeometry(side, side, side),
    overlayMaterials[layer].fill,
    grid.voxels.length,
  );
  cubes.castShadow = true;
  cubes.receiveShadow = true;

  const matrix = new THREE.Matrix4();
  const [cx, cy, cz] = grid.center;
  const [ox, oy, oz] = grid.origin;
  for (const [index, [x, y, z]] of grid.voxels.entries()) {
    matrix.setPosition(
      cx + ox + (x + 0.5) * grid.voxelSize,
      cy + oy + (y + 0.5) * grid.voxelSize,
      cz + oz + (z + 0.5) * grid.voxelSize,
    );
    cubes.setMatrixAt(index, matrix);
  }
  group.add(cubes);
  return group;
}

/**
 * Fetches (once per run) the voxel geometry the placement model reads and attaches the
 * current mode's rendering to the loaded meshes. Parenting each overlay to the mesh's
 * `placement` node, when it has one, hands it the baked pose for free — the cubes follow
 * the mesh wherever it was placed.
 */
async function showOverlay(id: string): Promise<void> {
  const mode = overlayMode;
  if (!mode) return;

  const key = `${corpus}/${id}@${currentRes()}`;
  let grids = gridsCache.get(key);
  if (!grids) {
    setStatus('Computing voxel geometry…', 'busy');
    const response = await fetch(`/api/blocks/${encodeURIComponent(id)}?res=${currentRes()}${corpusQuery()}`);
    const body = (await response.json()) as (Record<Layer, Grid> & { error?: string }) | { error: string };
    if (!response.ok || !('anchor' in body)) throw new Error(body.error ?? `HTTP ${response.status}`);
    grids = body;
    gridsCache.set(key, grids);
  }

  for (const layer of LAYERS) {
    const host = layers[layer];
    if (!host || overlays[layer]) continue;
    const overlay = mode === 'blocks' ? buildBlocksOverlay(layer, grids[layer]) : buildVoxelsOverlay(layer, grids[layer]);
    overlays[layer] = overlay;
    (host.getObjectByName('placement') ?? host).add(overlay);
  }

  const counts = (pick: (grid: Grid) => number) => `${pick(grids.anchor)} anchor + ${pick(grids.placed)} placed`;
  setStatus(
    mode === 'blocks'
      ? `Blocks: ${counts((grid) => grid.blocks.length)}.`
      : `Voxels: ${counts((grid) => grid.voxels.length)}.`,
    'ok',
  );
}

/**
 * Parks a raw pair side by side: each object centred on X/Z, rested on the grid, and pushed
 * apart along X far enough to clear both bounding boxes. Only used when the loaded files
 * carry no baked pose — posed files are shown exactly as they say, untouched.
 */
function separateLayers(): void {
  const boxes = LAYERS.map((layer) => new THREE.Box3().setFromObject(layers[layer]!));
  const sizes = boxes.map((box) => box.getSize(new THREE.Vector3()));
  const gap = Math.max(...sizes.map((size) => size.length())) * 0.06;

  for (const [i, layer] of LAYERS.entries()) {
    const centre = boxes[i].getCenter(new THREE.Vector3());
    const direction = i === 0 ? -1 : 1;
    const object = layers[layer]!;
    object.position.x += direction * (sizes[i].x / 2 + gap) - centre.x;
    object.position.y += -boxes[i].min.y;
    object.position.z += -centre.z;
  }
}

const dims = (size: Vec3) => size.map((v) => v.toFixed(3)).join(' × ');

function renderInfo(sample: Sample, base: string): void {
  dom.placement.textContent = `"${sample.placement}"`;

  dom.selInfo.textContent = '';
  for (const layer of LAYERS) {
    const badge = document.createElement('span');
    badge.className = `tag ${TAGS[layer]}`;
    badge.textContent = layer;
    dom.selInfo.append(badge, ` ${sample[layer].name}\u00a0\u00a0`);
  }

  dom.numbers.textContent = '';
  const entries: [string, string][] = [
    ['context', sample.context],
    ['phrasing', sample.phrasing],
    ['anchor size', dims(sample.anchor.size)],
    ['placed size', dims(sample.placed.size)],
    ['combined', dims(sample.combined_size)],
  ];
  for (const [key, value] of entries) {
    const row = document.createElement('div');
    row.className = 'bbox-row';
    const label = document.createElement('span');
    label.textContent = key;
    const text = document.createElement('span');
    text.textContent = value;
    row.append(label, text);
    dom.numbers.appendChild(row);
  }

  dom.descriptions.textContent = '';
  for (const layer of LAYERS) {
    const part = sample[layer];
    const card = document.createElement('div');
    card.className = 'part';

    if (part.image) {
      const image = document.createElement('img');
      image.src = `${base}/${encodeURIComponent(part.image)}`;
      image.alt = part.name;
      image.loading = 'lazy';
      card.appendChild(image);
    }

    const text = document.createElement('p');
    text.className = 'placement';
    text.textContent = `${part.name} — ${part.description}`;
    card.appendChild(text);

    dom.descriptions.appendChild(card);
  }
}

/** One placement's full model conversation, as `/api/placement-logs` parses it out of the
 *  audit log: the exact system and user prompt the model saw, its answer, and what the
 *  physics pass did. `current` marks the log whose baked transforms are the pose on screen. */
interface PlacementLog {
  file: string;
  stamp: string | null;
  model: string | null;
  tokens: string | null;
  cost: string | null;
  placement: string;
  system: string;
  user: string;
  answer: string;
  physics: string;
  transforms: unknown;
  current?: boolean;
}

let llmLogs: PlacementLog[] = [];

function clearLlmPanel(hint = ''): void {
  llmLogs = [];
  dom.llmSelect.innerHTML = '';
  dom.llmLog.textContent = '';
  dom.llmHint.textContent = hint;
}

function renderLlmLog(log: PlacementLog): void {
  dom.llmLog.textContent = '';

  const meta = document.createElement('p');
  meta.className = 'llm-meta';
  meta.textContent = [
    log.model ?? 'physics-only pass — no model call',
    log.tokens,
    log.cost,
    log.current === false ? 'not the pose on screen' : null,
  ].filter(Boolean).join('\n');
  dom.llmLog.appendChild(meta);

  const section = (title: string, text: string, open = false) => {
    if (!text) return;
    const details = document.createElement('details');
    details.open = open;
    const summary = document.createElement('summary');
    summary.textContent = title;
    const pre = document.createElement('pre');
    pre.textContent = text;
    details.append(summary, pre);
    dom.llmLog.appendChild(details);
  };

  section('Answer', log.answer, true);
  section('User prompt (LLM input)', log.user);
  section('System prompt', log.system);
  section('Physics', log.physics);
  if (log.transforms) section('Baked transforms', JSON.stringify(log.transforms, null, 2));
}

/** Fills the LLM panel with every placement log this sample has, newest first, preselecting
 *  the one whose transforms match the currently baked pose. */
async function loadLlmLogs(id: string): Promise<void> {
  clearLlmPanel('Loading placement logs…');
  try {
    const response = await fetch(
      `/api/placement-logs/${encodeURIComponent(id)}?${corpusQuery().slice(1)}`,
      { cache: 'no-store' },
    );
    const body = (await response.json()) as { logs?: PlacementLog[]; error?: string };
    if (!response.ok || !body.logs) throw new Error(body.error ?? `HTTP ${response.status}`);

    llmLogs = body.logs;
    if (!llmLogs.length) {
      dom.llmHint.textContent = 'No placement logs for this sample — nothing has placed it from this machine.';
      return;
    }

    for (const [index, log] of llmLogs.entries()) {
      const option = document.createElement('option');
      option.value = String(index);
      const when = log.stamp ? new Date(log.stamp).toLocaleString() : log.file;
      option.textContent = `${log.current ? '● ' : ''}${when} — ${log.model ?? 'physics only'}`;
      dom.llmSelect.appendChild(option);
    }

    const matched = llmLogs.findIndex((log) => log.current);
    const chosen = Math.max(matched, 0);
    dom.llmSelect.value = String(chosen);
    renderLlmLog(llmLogs[chosen]);
    dom.llmHint.textContent = matched >= 0
      ? ''
      : 'No log matches the baked pose (placed elsewhere or restored) — showing the newest.';
  } catch (err) {
    clearLlmPanel(`Logs unavailable: ${(err as Error).message}`);
  }
}

dom.llmSelect.addEventListener('change', () => {
  const log = llmLogs[Number(dom.llmSelect.value)];
  if (log) renderLlmLog(log);
});

const UUID_SUFFIX = /_[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i;

/** `coffee_mug_2b4bef1f-….glb` → `coffee mug`, the only name a spec.json folder gives up. */
const meshLabel = (mesh: string) => mesh.replace(/\.glb$/i, '').replace(UUID_SUFFIX, '').replace(/_/g, ' ');

/** Blanks every readout a placed sample fills, since a spec.json folder answers none of them. */
function renderFiles(run: Run): void {
  dom.placement.textContent = 'No metadata.json — nothing is recorded about this pair.';
  dom.numbers.textContent = '';

  dom.selInfo.textContent = '';
  dom.descriptions.textContent = '';
  for (const [i, layer] of LAYERS.entries()) {
    const badge = document.createElement('span');
    badge.className = `tag ${TAGS[layer]}`;
    badge.textContent = layer;
    dom.selInfo.append(badge, ` ${meshLabel(run.meshes[i])}\u00a0\u00a0`);

    const row = document.createElement('p');
    row.className = 'placement';
    row.textContent = run.meshes[i];
    dom.descriptions.appendChild(row);
  }
}

async function loadRun(id: string): Promise<void> {
  const run = runs.find((entry) => entry.id === id);
  if (!run) return;

  clearLayers();
  setStatus(`Loading ${id}…`, 'busy');
  const base = sampleBase(id);

  try {
    // Served out of the sample folder when its meshes are there, off the scene volume when they
    // are not. Either way, whatever pose a mesh has is already inside the file. The cache-buster
    // matters after "Place with LLM": the same URL must yield the freshly posed mesh. `raw`
    // bypasses the posed copies — that is the Reset view.
    const bust = `v=${Date.now()}${showRaw ? '&raw=1' : ''}${corpusQuery()}`;
    const loaded = await Promise.all(
      run.meshes.map((mesh) => loader.loadAsync(`/mesh/${encodeURIComponent(id)}/${encodeURIComponent(mesh)}?${bust}`)),
    );
    for (const [i, layer] of LAYERS.entries()) {
      layers[layer] = loaded[i].scene;
      scene.add(loaded[i].scene);
    }

    // Only a file with a baked `placement` node is actually posed; a pair of raw meshes
    // would land on top of each other, so those are parked apart instead.
    const posed = LAYERS.every((layer) => layers[layer]!.getObjectByName('placement'));
    if (!posed) separateLayers();
    applyVisibility();

    if (run.placed) {
      const response = await fetch(`${base}/metadata.json`, { cache: 'no-store' });
      if (!response.ok) throw new Error(`metadata.json: HTTP ${response.status}`);
      renderInfo((await response.json()) as Sample, base);
    } else {
      renderFiles(run);
    }
    void loadLlmLogs(id);

    frame();
    setStatus(
      posed
        ? 'Loaded posed GLBs — the arrangement comes only from the files.'
        : 'Loaded raw meshes — nothing has posed this pair yet, so the two are shown apart.',
      'ok',
    );

    if (overlayMode) {
      await showOverlay(id).catch((err) => setStatus(`Overlay unavailable: ${(err as Error).message}`, 'err'));
    }
  } catch (err) {
    setStatus(`Load failed: ${(err as Error).message}`, 'err');
  }
}

async function loadRuns(): Promise<void> {
  try {
    const response = await fetch(`/api/runs?${corpusQuery().slice(1)}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    runs = ((await response.json()) as { runs: Run[] }).runs ?? [];

    const previous = dom.runSelect.value;
    dom.runSelect.innerHTML = '<option value="">— select run —</option>';
    for (const run of runs) {
      const option = document.createElement('option');
      option.value = run.id;
      option.textContent = run.placed ? run.id : `${run.id} (unposed)`;
      dom.runSelect.appendChild(option);
    }
    if (runs.some((run) => run.id === previous)) dom.runSelect.value = previous;

    const placed = runs.filter((run) => run.placed).length;
    dom.runHint.textContent = runs.length
      ? `${placed} placed, ${runs.length - placed} unposed.`
      : 'Nothing to show yet — try `npm run pipeline`.';
  } catch (err) {
    dom.runHint.textContent = `Failed to list runs: ${(err as Error).message}`;
  }
}

/** Fills the corpus picker: every `generated*` folder the server can see, sample counts
 *  along for orientation. The URL's choice survives a reload; otherwise the server's
 *  default corpus is selected. */
async function loadCorpora(): Promise<void> {
  try {
    const response = await fetch('/api/corpora');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const body = (await response.json()) as { corpora: { id: string; samples: number }[]; default: string };

    dom.corpusSelect.innerHTML = '';
    if (!body.corpora.length) {
      dom.corpusSelect.innerHTML = '<option value="">default</option>';
    }
    for (const entry of body.corpora) {
      const option = document.createElement('option');
      option.value = entry.id;
      option.textContent = `${entry.id} (${entry.samples})`;
      dom.corpusSelect.appendChild(option);
    }

    const wanted = initialParams.get('corpus');
    corpus = body.corpora.some((entry) => entry.id === wanted) ? wanted! : body.default;
    dom.corpusSelect.value = corpus;
  } catch {
    dom.corpusSelect.innerHTML = '<option value="">default</option>';
    corpus = '';
  }
}

/** The part of a placement log worth reading back: the sample's own `▸` block. */
const answerTail = (log = '') => {
  const at = log.indexOf('▸');
  return (at >= 0 ? log.slice(at) : log).trim().slice(0, 600);
};

const MODEL_STORE = 'placement-model';
let modelCatalog: { id: string }[] = [];

// Captured before anything rewrites the URL: restoring run and model races the two fetches,
// and whichever finishes first would otherwise sync the URL and erase the other's parameter.
const initialParams = new URLSearchParams(location.search);
if (initialParams.get('res')) dom.resInput.value = initialParams.get('res')!;
if (initialParams.get('reasoning')) dom.reasoningSelect.value = initialParams.get('reasoning')!;
if (initialParams.get('contact')) dom.contactSelect.value = initialParams.get('contact')!;

/** The voxel resolution the controls ask for, clamped to what the pipeline takes. */
function currentRes(): number {
  const n = Math.round(Number(dom.resInput.value));
  return Number.isFinite(n) && n >= 4 && n <= 128 ? n : 16;
}

/** Rebuilds the model options to those matching the search box, keeping the choice if it survives. */
function renderModelOptions(): void {
  const query = dom.modelSearch.value.trim().toLowerCase();
  const matches = query ? modelCatalog.filter((model) => model.id.toLowerCase().includes(query)) : modelCatalog;
  const chosen = dom.modelSelect.value;

  dom.modelSelect.innerHTML = '';
  for (const model of matches) {
    const option = document.createElement('option');
    option.value = model.id;
    option.textContent = model.id;
    dom.modelSelect.appendChild(option);
  }
  if (matches.some((model) => model.id === chosen)) dom.modelSelect.value = chosen;
}

/**
 * Fills the model picker from OpenRouter's catalog (already filtered server-side to models
 * that can answer a structured-output call). The URL beats localStorage beats the pipeline
 * default; a picker that fails to load degrades to the default rather than blocking placement.
 */
async function loadModels(): Promise<void> {
  try {
    const response = await fetch('/api/placement-models');
    const body = (await response.json()) as { models?: { id: string }[]; default?: string; error?: string };
    if (!response.ok || !body.models?.length) throw new Error(body.error ?? `HTTP ${response.status}`);

    modelCatalog = body.models;
    renderModelOptions();
    const wanted = [initialParams.get('model'), localStorage.getItem(MODEL_STORE), body.default]
      .find((id) => id && modelCatalog.some((model) => model.id === id));
    if (wanted) dom.modelSelect.value = wanted;
    syncUrl();
  } catch {
    dom.modelSelect.innerHTML = '<option value="">pipeline default</option>';
  }
}

/** Mirrors the restorable state — which run and which model — into the URL, so a reload
 *  (or a shared link) lands back on the same view. */
function syncUrl(): void {
  const params = new URLSearchParams(location.search);
  const entries: [string, string][] = [
    ['corpus', corpus],
    ['run', dom.runSelect.value],
    ['model', dom.modelSelect.value],
    ['res', String(currentRes())],
    ['reasoning', dom.reasoningSelect.value],
    ['contact', dom.contactSelect.value],
  ];
  for (const [key, value] of entries) {
    if (value) params.set(key, value);
    else params.delete(key);
  }
  history.replaceState(null, '', `${location.pathname}?${params}`);
}

/** Shows the pair in its original, un-placed arrangement: raw meshes, parked side by side. */
async function resetView(): Promise<void> {
  showRaw = true;
  const id = dom.runSelect.value;
  if (id) await loadRun(id);
}

/** Which of the placement pipeline's stages a click asks the server for. */
type PlaceMode = 'place' | 'physics' | 'full';

const MODE_STATUS: Record<PlaceMode, (id: string) => string> = {
  place: (id) => `Placing ${id} — voxelize, LLM, bake (no physics)…`,
  physics: (id) => `Applying physics to ${id} — refine against the meshes, re-bake…`,
  full: (id) => `Placing ${id} — voxelize, LLM, physics, bake…`,
};

function setRunControls(enabled: boolean): void {
  for (const button of [dom.placeRun, dom.physicsRun, dom.fullRun]) button.disabled = !enabled;
}

/** Runs one of the placement modes through the server, then reloads the freshly posed files. */
async function placeRun(mode: PlaceMode): Promise<void> {
  const id = dom.runSelect.value;
  if (!id) return;

  setRunControls(false);
  dom.placeLog.textContent = '';
  clearGhost();
  // An LLM pass snaps back to the original arrangement first, so the wait shows the
  // before-state and the reload afterwards shows exactly what the placement changed.
  // Physics continues from the pose on screen, so the screen keeps showing it.
  if (mode !== 'physics') await resetView();
  setStatus(MODE_STATUS[mode](id), 'busy');
  try {
    const response = await fetch('/api/place', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        id,
        mode,
        corpus: corpus || undefined,
        model: dom.modelSelect.value || undefined,
        resolution: currentRes(),
        reasoning: dom.reasoningSelect.value || undefined,
        contact: dom.contactSelect.value || undefined,
      }),
    });
    const result = (await response.json()) as { ok?: boolean; log?: string; error?: string; ghost?: GhostTRS };
    if (!response.ok || !result.ok) throw new Error(result.error ?? (answerTail(result.log) || `HTTP ${response.status}`));

    dom.placeLog.textContent = answerTail(result.log);
    showRaw = false;
    await loadRun(id);
    if (result.ghost) await showGhost(result.ghost);
  } catch (err) {
    setStatus(`${mode === 'physics' ? 'Physics' : 'Placement'} failed: ${(err as Error).message}`, 'err');
  } finally {
    setRunControls(!!dom.runSelect.value);
  }
}

for (const key of LAYERS) {
  buttons[key].addEventListener('click', () => {
    visible[key] = !visible[key];
    applyVisibility();
  });
}

dom.placeRun.addEventListener('click', () => void placeRun('place'));
dom.physicsRun.addEventListener('click', () => void placeRun('physics'));
dom.fullRun.addEventListener('click', () => void placeRun('full'));

const applyBlockOpacity = () => {
  const opacity = Number(dom.blockOpacity.value) / 100;
  for (const layer of LAYERS) {
    const { fill } = overlayMaterials[layer];
    fill.opacity = opacity;
    // See-through blending wants depth writes off; a near-solid overlay wants them on, or
    // its own hidden faces bleed through and the structure turns to mush.
    fill.depthWrite = opacity > 0.5;
  }
};
dom.blockOpacity.addEventListener('input', applyBlockOpacity);
applyBlockOpacity();

function setOverlayMode(mode: OverlayMode | null): void {
  overlayMode = mode;
  dom.toggleBlocks.classList.toggle('active', mode === 'blocks');
  dom.toggleVoxels.classList.toggle('active', mode === 'voxels');
  clearOverlays();

  const id = dom.runSelect.value;
  if (mode && id) {
    void showOverlay(id).catch((err) => {
      setOverlayMode(null);
      setStatus(`Overlay unavailable: ${(err as Error).message}`, 'err');
    });
  }
}

dom.toggleBlocks.addEventListener('click', () => setOverlayMode(overlayMode === 'blocks' ? null : 'blocks'));
dom.toggleVoxels.addEventListener('click', () => setOverlayMode(overlayMode === 'voxels' ? null : 'voxels'));
dom.toggleGhost.addEventListener('click', () => {
  if (!ghostObject) return;
  ghostVisible = !ghostVisible;
  ghostObject.visible = ghostVisible;
  dom.toggleGhost.classList.toggle('active', ghostVisible);
});

dom.reloadRuns.addEventListener('click', () => void loadRuns());
dom.frameView.addEventListener('click', frame);
dom.resetRun.addEventListener('click', () => {
  clearGhost();
  void resetView();
});
dom.modelSearch.addEventListener('input', renderModelOptions);
dom.resInput.addEventListener('change', () => {
  syncUrl();
  // An active overlay is showing the old resolution — rebuild it at the new one.
  if (overlayMode) setOverlayMode(overlayMode);
});
dom.reasoningSelect.addEventListener('change', syncUrl);
dom.contactSelect.addEventListener('change', syncUrl);
dom.modelSelect.addEventListener('change', () => {
  localStorage.setItem(MODEL_STORE, dom.modelSelect.value);
  syncUrl();
});

/** Reflects the run picker's current value on screen. `broadcast` tells the compare page's
 *  parent frame about the change, so a linked twin pane can follow — and is off exactly
 *  when the change *came* from the twin, which is what keeps two linked panes from
 *  ping-ponging forever. */
function applyRunSelection(broadcast: boolean): void {
  const id = dom.runSelect.value;
  setRunControls(!!id);
  dom.resetRun.disabled = !id;
  dom.placeLog.textContent = '';
  clearGhost();
  showRaw = false;
  syncUrl();
  if (broadcast) broadcastRun();
  if (id) void loadRun(id);
  else {
    clearLayers();
    applyVisibility();
    clearLlmPanel();
    setStatus('');
  }
}

dom.runSelect.addEventListener('change', () => applyRunSelection(true));

/** Tells the parent frame (the compare page) which run this pane is on. */
function broadcastRun(): void {
  if (window.parent === window) return;
  window.parent.postMessage({ type: 'pipeline-run', run: dom.runSelect.value }, location.origin);
}

// The compare page pushes the other pane's run here when its panes are linked. A run this
// corpus does not contain is ignored rather than guessed at.
window.addEventListener('message', (event) => {
  if (event.origin !== location.origin) return;
  const data = event.data as { type?: string; run?: string };
  if (data?.type !== 'pipeline-set-run' || typeof data.run !== 'string') return;
  if (data.run === dom.runSelect.value || !runs.some((run) => run.id === data.run)) return;
  dom.runSelect.value = data.run;
  applyRunSelection(false);
});

// Switching corpus swaps the whole sample list; whatever was on screen belongs to the old
// one, so the stage clears rather than showing a pair the new list may not even contain.
dom.corpusSelect.addEventListener('change', () => {
  corpus = dom.corpusSelect.value;
  dom.runSelect.value = '';
  setRunControls(false);
  dom.resetRun.disabled = true;
  dom.placeLog.textContent = '';
  clearGhost();
  clearLayers();
  applyVisibility();
  clearLlmPanel();
  setStatus('');
  syncUrl();
  void loadRuns();
});

// Compare: two of this page side by side, each pane its own viewer seeded with the
// current corpus — pick a different one in either pane from there.
dom.splitView.addEventListener('click', () => {
  const seed = corpus ? `?left=${encodeURIComponent(corpus)}&right=${encodeURIComponent(corpus)}` : '';
  location.href = `/compare.html${seed}`;
});

// Inside a compare pane the page is already split — hide the ways out of the frame.
if (initialParams.get('embed') === '1') {
  dom.splitView.style.display = 'none';
  document.getElementById('navLinks')?.style.setProperty('display', 'none');
}

// ---- sidebar sizing: drag the stage's edges, or collapse a side entirely --------------

interface Pane {
  aside: HTMLElement;
  handle: HTMLElement;
  toggle: HTMLButtonElement;
  width: number;
  open: boolean;
  min: number;
  max: number;
  /** Pointer x → wanted width, which is what differs between the two sides. */
  widthAt: (x: number) => number;
}

const appGrid = el<HTMLDivElement>('app');
const panes: Record<'left' | 'right', Pane> = {
  left: {
    aside: el('sidebar'),
    handle: el('handleLeft'),
    toggle: el<HTMLButtonElement>('toggleLeft'),
    width: 280,
    open: true,
    min: 200,
    max: 560,
    widthAt: (x) => x,
  },
  right: {
    aside: el('inspector'),
    handle: el('handleRight'),
    toggle: el<HTMLButtonElement>('toggleRight'),
    width: 320,
    open: true,
    min: 220,
    max: 640,
    widthAt: (x) => window.innerWidth - x,
  },
};

const PANE_STORE = 'pipeline-panes';
try {
  const stored = JSON.parse(localStorage.getItem(PANE_STORE) ?? '{}') as
    Partial<Record<'left' | 'right', { width: number; open: boolean }>>;
  for (const side of ['left', 'right'] as const) {
    const saved = stored[side];
    if (!saved) continue;
    if (Number.isFinite(saved.width)) panes[side].width = Math.min(Math.max(saved.width, panes[side].min), panes[side].max);
    panes[side].open = saved.open !== false;
  }
} catch {
  // unreadable state is just default widths
}

function applyPanes(): void {
  for (const side of ['left', 'right'] as const) {
    const pane = panes[side];
    pane.aside.classList.toggle('collapsed', !pane.open);
    pane.handle.style.display = pane.open ? '' : 'none';
    pane.toggle.textContent = (side === 'left') === pane.open ? '⟨' : '⟩';
  }
  appGrid.style.gridTemplateColumns =
    `${panes.left.open ? panes.left.width : 0}px 1fr ${panes.right.open ? panes.right.width : 0}px`;
  localStorage.setItem(PANE_STORE, JSON.stringify({
    left: { width: panes.left.width, open: panes.left.open },
    right: { width: panes.right.width, open: panes.right.open },
  }));
}

for (const side of ['left', 'right'] as const) {
  const pane = panes[side];
  pane.toggle.addEventListener('click', () => {
    pane.open = !pane.open;
    applyPanes();
  });
  pane.handle.addEventListener('pointerdown', (down) => {
    down.preventDefault();
    pane.handle.setPointerCapture(down.pointerId);
    pane.handle.classList.add('dragging');
    const move = (event: PointerEvent) => {
      pane.width = Math.min(Math.max(pane.widthAt(event.clientX), pane.min), pane.max);
      applyPanes();
    };
    const up = () => {
      pane.handle.classList.remove('dragging');
      pane.handle.removeEventListener('pointermove', move);
      pane.handle.removeEventListener('pointerup', up);
    };
    pane.handle.addEventListener('pointermove', move);
    pane.handle.addEventListener('pointerup', up);
  });
}
applyPanes();

window.addEventListener('keydown', (event) => {
  const target = event.target as HTMLElement;
  if (target?.tagName === 'INPUT' || target?.tagName === 'TEXTAREA' || target?.tagName === 'SELECT') return;
  if (event.key.toLowerCase() === 'f') frame();
});

applyVisibility();
void loadModels();
void loadCorpora()
  .then(() => loadRuns())
  .then(() => {
    const id = initialParams.get('run') ?? runs[0]?.id;
    if (id && runs.some((run) => run.id === id)) {
      dom.runSelect.value = id;
      setRunControls(true);
      dom.resetRun.disabled = false;
      syncUrl();
      broadcastRun();
      void loadRun(id);
    }
  });
