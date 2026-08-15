// A tile per LLM over one scene and one instruction — see pipeline/scene-edit.mjs for what a
// run is and where the answers come from.
//
// There is one WebGL context for the whole grid, not one per tile: each tile is a hole in a
// layer over a single canvas, and a frame walks the tiles setting the scissor box to each one
// in turn. A context per tile would upload its own copy of every geometry and texture in the
// room — eight tiles on a 13 MB room is eight times the video memory, and browsers start
// evicting contexts past sixteen. Sharing one, and sharing the geometry between the tile
// scenes, keeps that at one copy however many tiles are open.
//
// Every tile is drawn with the same camera, since six variants seen from six angles compare
// nothing. Framing comes from the run's own bounds rather than from what a tile contains, so a
// model that answers with a coordinate a hundred times too large leaves the frame instead of
// wrecking it.
//
// A run's objects are positioned entirely from the manifest: each mesh is offset by the centre
// recorded there inside a pivot placed at that same centre, so an object nothing touched lands
// exactly where the file has it, and rotation and scale happen about its own centre without
// anything here doing arithmetic on a model's answer.

import './style.css';
import './navigation';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { addStudioLighting } from './viewer';

type Vec3 = [number, number, number];
type Quat = [number, number, number, number];
type Status = 'idle' | 'running' | 'done' | 'failed' | 'stale';

interface SceneObject {
  id: string;
  source_id: string;
  label: string;
  node: string;
  index: number;
  center: Vec3;
  size: Vec3;
  on_floor: boolean;
  structural: boolean;
  facing_axis: '+X' | '-X' | '+Z' | '-Z' | null;
}

interface Run {
  id: string;
  prompt: string;
  prompt_hash: string;
  scene: { file: string; unit_scale: number; bounds: { min: Vec3; max: Vec3 }; floor_y: number };
  objects: SceneObject[];
}

/** One scene-API call exactly as a model returned it. Exactly one member is non-null. */
interface SceneCall {
  move_object: { id: string; center: Vec3 } | null;
  rotate_object: { id: string; yaw_degrees: number } | null;
  scale_object: { id: string; factor: number } | null;
  delete_object: { id: string } | null;
}

/** The same edit in normalised units, ready to assign to an object. */
interface Resolved {
  id: string;
  action: 'transform' | 'delete';
  origin: Vec3;
  position?: Vec3;
  rotation?: Quat;
  scale?: Vec3;
}

interface Result {
  model: string;
  latency_ms?: number;
  usage?: { cost?: number } | null;
  stale?: boolean;
  answer: { reasoning: string; calls: SceneCall[] };
  resolved: Resolved[];
  rejected: { index: number; name: string | null; id: string | null; why: string }[];
}

interface Job {
  model: string;
  status: 'running' | 'done' | 'failed';
  error: string | null;
}

interface Detail {
  run: Run;
  prompt: { system: string; user: string };
  results: Result[];
  jobs: Job[];
}

interface Listing {
  id: string;
  prompt: string;
  scene: string;
  objects: number;
  models: { model: string; stale: boolean }[];
}

interface SceneInfo {
  file: string;
  identified: boolean;
  objects: {
    id: string;
    label: string;
    size: Vec3;
    structural: boolean;
    facing_axis: '+X' | '-X' | '+Z' | '-Z' | null;
  }[];
  job: { status: 'running' | 'done' | 'failed'; stage: string; error: string | null } | null;
  error?: string;
}

interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  description: string;
  context_length: number | null;
  pricing: { prompt: number | null; completion: number | null };
  structured_outputs: boolean;
  response_format: boolean;
  created: number | null;
  expiration_date: string | null;
}

/** One room object, loaded once per run and shared by every tile. */
interface Source {
  mesh: THREE.Mesh;
  material: THREE.Material;
  center: THREE.Vector3;
  edges: THREE.BufferGeometry;
}

interface TileObject {
  pivot: THREE.Group;
  mesh: THREE.Mesh;
  box: THREE.LineSegments;
  before: THREE.LineSegments;
}

interface Tile {
  key: string;
  model: string | null;
  color: number;
  root: HTMLElement;
  head: HTMLElement;
  view: HTMLElement;
  detail: HTMLElement;
  scene: THREE.Scene;
  objects: Map<string, TileObject>;
  tint: THREE.MeshStandardMaterial;
  lines: THREE.LineBasicMaterial;
  owned: { dispose(): void }[];
  result: Result | null;
  status: Status;
  error: string | null;
}

const ORIGINAL = '\u0000original';
const PALETTE = [0xff8f5e, 0x5ec8ff, 0x3fb950, 0xd2a8ff, 0xffd166, 0xf85149];
const NEUTRAL = 0x8b949e;
const hex = (color: number) => `#${color.toString(16).padStart(6, '0')}`;

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node as T;
}

const dom = {
  stage: el<HTMLElement>('stage'),
  tiles: el<HTMLDivElement>('tiles'),
  runSelect: el<HTMLSelectElement>('runSelect'),
  rescan: el<HTMLButtonElement>('rescan'),
  runHint: el<HTMLParagraphElement>('runHint'),
  promptInput: el<HTMLTextAreaElement>('promptInput'),
  savePrompt: el<HTMLButtonElement>('savePrompt'),
  promptHint: el<HTMLParagraphElement>('promptHint'),
  promptText: el<HTMLPreElement>('promptText'),
  togglePrompt: el<HTMLButtonElement>('togglePrompt'),
  modelPicker: el<HTMLDivElement>('modelPicker'),
  modelControl: el<HTMLDivElement>('modelControl'),
  modelChips: el<HTMLDivElement>('modelChips'),
  modelSearch: el<HTMLInputElement>('modelSearch'),
  modelMenu: el<HTMLDivElement>('modelMenu'),
  modelHint: el<HTMLParagraphElement>('modelHint'),
  addViews: el<HTMLButtonElement>('addViews'),
  runDefault: el<HTMLButtonElement>('runDefault'),
  objects: el<HTMLDivElement>('objects'),
  toggleTextured: el<HTMLButtonElement>('toggleTextured'),
  toggleTint: el<HTMLButtonElement>('toggleTint'),
  toggleBoxes: el<HTMLButtonElement>('toggleBoxes'),
  frameView: el<HTMLButtonElement>('frameView'),
  planView: el<HTMLButtonElement>('planView'),
  columns: el<HTMLSelectElement>('columns'),
  sceneSelect: el<HTMLSelectElement>('sceneSelect'),
  createRun: el<HTMLButtonElement>('createRun'),
  status: el<HTMLParagraphElement>('status'),
};

// ── the one renderer ─────────────────────────────────────────────────────────

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setClearColor(0x0d1117, 1);
dom.stage.insertBefore(renderer.domElement, dom.tiles);

const environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 5000);
camera.position.set(1.2, 1, 2.4);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
orbit.dampingFactor = 0.08;

const loader = new GLTFLoader();
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

/** Shared: the untextured look, and the muted box marking where an edited object came from. */
const FLAT = new THREE.MeshStandardMaterial({ color: 0x9aa4b2, roughness: 0.6, metalness: 0.02 });
const BEFORE = new THREE.LineBasicMaterial({ color: 0x6e7681, transparent: true, opacity: 0.7 });

// ── state ────────────────────────────────────────────────────────────────────

let listings: Listing[] = [];
let sceneCatalogue: SceneInfo[] = [];
let scenePoller: number | null = null;
let defaultModels: string[] = [];
let modelResults: ModelInfo[] = [];
let selectedModels: ModelInfo[] = [];
let modelSearchTimer: number | null = null;
let modelSearchRequest = 0;
let activeModelIndex = -1;
let run: Run | null = null;
const sources = new Map<string, Source>();
const tiles: Tile[] = [];
let highlight: string | null = null;
let focus: string | null = null;
let columns = 0;
let textured = true;
let tinted = false;
let showBoxes = true;
let poller: number | null = null;
let canvasSize = { width: 0, height: 0 };

function setStatus(text: string, kind: '' | 'ok' | 'err' | 'busy' = ''): void {
  dom.status.textContent = text;
  dom.status.className = `status ${kind}`.trim();
}

// ── drawing ──────────────────────────────────────────────────────────────────

(function draw(): void {
  requestAnimationFrame(draw);
  orbit.update();

  const stage = dom.stage.getBoundingClientRect();
  if (stage.width !== canvasSize.width || stage.height !== canvasSize.height) {
    canvasSize = { width: stage.width, height: stage.height };
    renderer.setSize(stage.width, stage.height, false);
  }
  if (!stage.width || !stage.height) return;

  // The gaps between tiles belong to no scissor box, so they are cleared once up front.
  renderer.setScissorTest(false);
  renderer.clear();
  renderer.setScissorTest(true);

  for (const tile of tiles) {
    const rect = tile.view.getBoundingClientRect();
    if (rect.width < 4 || rect.height < 4) continue;
    const x = Math.round(rect.left - stage.left);
    const y = Math.round(stage.bottom - rect.bottom);
    const width = Math.round(rect.width);
    const height = Math.round(rect.height);
    renderer.setViewport(x, y, width, height);
    renderer.setScissor(x, y, width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.render(tile.scene, camera);
  }
})();

/** Mostly along +Z, which is the view the prompt tells every model "left" and "right" are in. */
const EYE = new THREE.Vector3(0.35, 0.5, 1);
/** Barely off the pole, since a camera exactly above its target has no defined roll. */
const PLAN = new THREE.Vector3(0, 1, 0.06);

/**
 * Always framed on the run's own bounds rather than on what a tile holds, so every tile keeps
 * the same camera and a model that answers with a coordinate a hundred times too large leaves
 * the frame instead of wrecking it.
 */
function frame(eye = EYE): void {
  if (!run) return;
  const scale = run.scene.unit_scale;
  const bounds = new THREE.Box3(
    new THREE.Vector3().fromArray(run.scene.bounds.min).divideScalar(scale),
    new THREE.Vector3().fromArray(run.scene.bounds.max).divideScalar(scale),
  );
  const sphere = bounds.getBoundingSphere(new THREE.Sphere());
  const direction = eye.clone().normalize();
  const distance = (sphere.radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.5;
  orbit.target.copy(sphere.center);
  camera.position.copy(sphere.center).addScaledVector(direction, Math.max(distance, 0.4));
  camera.near = Math.max(distance / 100, 0.001);
  camera.far = distance * 100;
  camera.updateProjectionMatrix();
  orbit.update();
}

// ── a run's objects ──────────────────────────────────────────────────────────

/** Everything that survives any loader's name sanitising, which is what makes this comparable. */
const loose = (name: string) => name.toLowerCase().replace(/[^a-z0-9]+/g, '');

function disposeMaterial(material: THREE.Material): void {
  for (const value of Object.values(material)) if (value instanceof THREE.Texture) value.dispose();
  material.dispose();
}

/**
 * Loads the room once and picks each object out of it by the child index the manifest recorded.
 * The index rather than the name, because a glTF loader sanitises node names — `…_1.glb` comes
 * back as `…_1obj` — and matching on a reimplementation of that rule would break the day the
 * rule changes. The name is still compared loosely, so a scene that is not the one the run was
 * measured from says so instead of silently mislabelling every object.
 */
async function buildSources(target: Run): Promise<void> {
  const gltf = await loader.loadAsync(`/scenes/${encodeURIComponent(target.scene.file)}`);
  const roots = gltf.scene.children;
  if (roots.length !== 1) throw new Error(`${target.scene.file}: expected one root node, found ${roots.length}`);

  const world = roots[0];
  if (world.children.length !== target.objects.length) {
    throw new Error(`${target.scene.file} holds ${world.children.length} objects, the run recorded ${target.objects.length}`);
  }

  for (const object of target.objects) {
    const node = world.children[object.index];
    if (!(node instanceof THREE.Mesh)) throw new Error(`${object.id}: child ${object.index} is not a mesh`);
    if (loose(node.name) !== loose(object.node)) {
      throw new Error(`${object.id}: the scene has "${node.name}" where the run recorded "${object.node}"`);
    }

    const size = new THREE.Vector3().fromArray(object.size).divideScalar(target.scene.unit_scale);
    const cube = new THREE.BoxGeometry(size.x, size.y, size.z);
    const edges = new THREE.EdgesGeometry(cube);
    cube.dispose();

    sources.set(object.id, {
      mesh: node,
      material: (Array.isArray(node.material) ? node.material[0] : node.material) as THREE.Material,
      center: new THREE.Vector3().fromArray(object.center).divideScalar(target.scene.unit_scale),
      edges,
    });
  }
}

function clearSources(): void {
  for (const source of sources.values()) {
    source.mesh.geometry.dispose();
    source.edges.dispose();
    disposeMaterial(source.material);
  }
  sources.clear();
}

// ── tiles ────────────────────────────────────────────────────────────────────

function colorOf(model: string): number {
  const known = defaultModels.indexOf(model);
  if (known >= 0) return PALETTE[known % PALETTE.length];
  let hash = 0;
  for (const character of model) hash = (hash * 31 + character.charCodeAt(0)) % 997;
  return PALETTE[hash % PALETTE.length];
}

function createTile(model: string | null): Tile {
  const color = model ? colorOf(model) : NEUTRAL;
  const owned: { dispose(): void }[] = [];

  const root = document.createElement('article');
  root.className = 'tile';
  const head = document.createElement('header');
  head.className = 'tile__head';
  const view = document.createElement('div');
  view.className = 'tile__view';
  const detail = document.createElement('div');
  detail.className = 'tile__detail';
  detail.hidden = true;
  root.append(head, view, detail);
  dom.tiles.appendChild(root);

  const scene = new THREE.Scene();
  scene.environment = environment;
  addStudioLighting(scene);

  if (run) {
    const { bounds, floor_y: floor, unit_scale: scale } = run.scene;
    const span = Math.max(bounds.max[0] - bounds.min[0], bounds.max[2] - bounds.min[2]) / scale;
    const grid = new THREE.GridHelper(span * 1.5, 12, 0x2a313c, 0x1c2230);
    const surface = grid.material as THREE.Material;
    surface.transparent = true;
    surface.opacity = 0.45;
    // The rooms carry no floor geometry, so the grid is drawn where the floor is measured to be.
    grid.position.y = floor / scale;
    const axes = new THREE.AxesHelper(span * 0.12);
    axes.position.y = floor / scale;
    scene.add(grid, axes);
    owned.push(grid.geometry, surface, axes.geometry, axes.material as THREE.Material);
  }

  const tint = new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.03 });
  const lines = new THREE.LineBasicMaterial({ color });
  owned.push(tint, lines);

  const objects = new Map<string, TileObject>();
  for (const [id, source] of sources) {
    const pivot = new THREE.Group();
    pivot.userData.objectId = id;
    pivot.position.copy(source.center);

    const mesh = source.mesh.clone();
    mesh.position.copy(source.center).negate();
    const box = new THREE.LineSegments(source.edges, lines);
    box.raycast = () => {};
    pivot.add(mesh, box);

    const before = new THREE.LineSegments(source.edges, BEFORE);
    before.position.copy(source.center);
    before.raycast = () => {};

    scene.add(pivot, before);
    objects.set(id, { pivot, mesh, box, before });
  }

  return {
    key: model ?? ORIGINAL,
    model,
    color,
    root,
    head,
    view,
    detail,
    scene,
    objects,
    tint,
    lines,
    owned,
    result: null,
    status: model ? 'idle' : 'done',
    error: null,
  };
}

function disposeTile(tile: Tile): void {
  tile.root.remove();
  for (const thing of tile.owned) thing.dispose();
  tile.objects.clear();
  tile.scene.clear();
}

function removeTile(tile: Tile): void {
  tiles.splice(tiles.indexOf(tile), 1);
  if (focus === tile.key) focus = null;
  disposeTile(tile);
  layout();
  ensurePolling();
}

function layout(): void {
  const shown = tiles.filter((tile) => focus === null || tile.key === focus);
  for (const tile of tiles) {
    tile.root.hidden = !shown.includes(tile);
    tile.root.classList.toggle('focus', focus === tile.key);
  }
  const count = Math.max(shown.length, 1);
  const wide = count <= 1 ? 1 : count <= 4 ? 2 : count <= 9 ? 3 : 4;
  const across = focus !== null ? 1 : columns || wide;
  dom.tiles.style.gridTemplateColumns = `repeat(${across}, minmax(0, 1fr))`;
  dom.tiles.style.gridTemplateRows = `repeat(${Math.ceil(count / across)}, minmax(0, 1fr))`;
}

/** Poses one tile's objects from its model's answer. Everything else stays as the file has it. */
function applyTile(tile: Tile): void {
  const edits = new Map((tile.result?.resolved ?? []).map((entry) => [entry.id, entry]));

  for (const [id, object] of tile.objects) {
    const source = sources.get(id);
    if (!source) continue;
    const edit = edits.get(id);
    const removed = edit?.action === 'delete';

    object.pivot.visible = !removed;
    if (edit?.action === 'transform' && edit.position && edit.rotation && edit.scale) {
      object.pivot.position.fromArray(edit.position);
      object.pivot.quaternion.fromArray(edit.rotation);
      object.pivot.scale.fromArray(edit.scale);
    } else {
      object.pivot.position.copy(source.center);
      object.pivot.quaternion.identity();
      object.pivot.scale.set(1, 1, 1);
    }

    object.box.visible = !removed && (highlight === id || (showBoxes && !!edit));
    // Where a moved object came from, and the only sign left of a deleted one.
    object.before.visible = showBoxes && !!edit;
    object.mesh.material = edit && tinted ? tile.tint : textured ? source.material : FLAT;
  }
}

// ── tile chrome ──────────────────────────────────────────────────────────────

function button(text: string, title: string, onClick: () => void): HTMLButtonElement {
  const element = document.createElement('button');
  element.className = 'ghost';
  element.textContent = text;
  element.title = title;
  element.addEventListener('click', onClick);
  return element;
}

function pillsFor(tile: Tile): [string, string][] {
  const pills: [string, string][] = [];
  if (tile.status === 'running') pills.push(['asking', 'busy']);
  if (tile.status === 'failed') pills.push(['failed', 'err']);
  if (tile.status === 'stale') pills.push(['stale', 'warn']);
  if (tile.result?.rejected.length) pills.push([`${tile.result.rejected.length} rejected`, 'warn']);
  if (tile.result && !tile.result.resolved.length) pills.push(['no changes', 'warn']);
  return pills;
}

function metaFor(tile: Tile): string {
  if (!tile.model) return `${sources.size} objects`;
  if (!tile.result) return '';
  const { answer, latency_ms: latency, usage } = tile.result;
  const parts = [`${answer.calls.length} call${answer.calls.length === 1 ? '' : 's'}`];
  if (latency) parts.push(`${(latency / 1000).toFixed(1)}s`);
  if (usage?.cost) parts.push(`$${usage.cost.toFixed(4)}`);
  return parts.join(' · ');
}

function describeCall(call: SceneCall, index: number): string {
  if (call.move_object) {
    return `${index + 1}. move_object(${call.move_object.id}, [${call.move_object.center.join(', ')}])`;
  }
  if (call.rotate_object) {
    return `${index + 1}. rotate_object(${call.rotate_object.id}, ${call.rotate_object.yaw_degrees}°)`;
  }
  if (call.scale_object) {
    return `${index + 1}. scale_object(${call.scale_object.id}, ×${call.scale_object.factor})`;
  }
  if (call.delete_object) return `${index + 1}. delete_object(${call.delete_object.id})`;
  return `${index + 1}. invalid call`;
}

function renderDetail(tile: Tile): void {
  tile.detail.textContent = '';
  if (!tile.model) return;

  const section = (title: string, body: string, muted = false): void => {
    const heading = document.createElement('h4');
    heading.textContent = title;
    const text = document.createElement('pre');
    text.textContent = body;
    if (muted) text.className = 'muted';
    tile.detail.append(heading, text);
  };

  if (tile.error) section('Failed', tile.error);
  if (!tile.result) {
    if (!tile.error) section('Waiting', 'Nothing back from this model yet.', true);
    return;
  }

  section('Reasoning', tile.result.answer.reasoning || '(none given)', true);
  const calls = tile.result.answer.calls;
  section('Calls', calls.length ? calls.map(describeCall).join('\n') : 'It made no calls.');
  if (tile.result.rejected.length) {
    section(
      'Rejected',
      tile.result.rejected
        .map((entry) => `#${entry.index + 1} ${entry.name ?? '(no call)'} · ${entry.id ?? '(no id)'} — ${entry.why}`)
        .join('\n'),
    );
  }
}

function renderChrome(tile: Tile): void {
  tile.head.textContent = '';

  const name = document.createElement('span');
  name.className = 'tile__name';
  const swatch = document.createElement('span');
  swatch.className = 'swatch';
  swatch.style.background = hex(tile.color);
  const label = document.createElement('span');
  // The part after the provider, since that is what distinguishes one tile from the next and a
  // tile can be a third of the stage wide. The full id is in the tooltip.
  label.textContent = tile.model?.split('/').pop() ?? 'Original';
  name.append(swatch, label);
  name.title = tile.model
    ? `${tile.model} — click to give it the whole grid`
    : 'The scene as the file has it — click to give it the whole grid';
  name.addEventListener('click', () => {
    focus = focus === tile.key ? null : tile.key;
    layout();
  });
  tile.head.appendChild(name);

  for (const [text, kind] of pillsFor(tile)) {
    const pill = document.createElement('span');
    pill.className = `pill ${kind}`;
    pill.textContent = text;
    tile.head.appendChild(pill);
  }

  const meta = document.createElement('span');
  meta.className = 'tile__meta';
  meta.textContent = metaFor(tile);
  tile.head.appendChild(meta);

  if (tile.model) {
    tile.head.append(
      button('info', 'Its reasoning and the calls it returned', () => {
        tile.detail.hidden = !tile.detail.hidden;
      }),
      button('↻', 'Ask this model again', () => void addView(tile.model as string, true)),
      button('✕', 'Close this view — the answer stays on disk', () => removeTile(tile)),
    );
  }

  renderDetail(tile);
}

// ── the sidebar ──────────────────────────────────────────────────────────────

function renderObjects(): void {
  dom.objects.textContent = '';
  if (!run) return;

  for (const object of run.objects) {
    const row = document.createElement('div');
    row.className = 'object-row';
    row.dataset.id = object.id;
    row.classList.toggle('active', highlight === object.id);
    row.title = `${object.id} · front ${object.facing_axis ?? 'none'} · centre ${object.center.join(', ')} · ${object.on_floor ? 'on the floor' : 'off the floor'}`;

    const label = document.createElement('input');
    label.value = object.label;
    label.spellcheck = false;
    label.disabled = object.structural;
    label.title = object.structural ? 'Structural scene geometry keeps its original ID and name' : 'What a model is told this object is called';
    // Focus doubles as the highlight, so the object being renamed is ringed while you type.
    label.addEventListener('focus', () => setHighlight(object.id));
    label.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') label.blur();
      else if (event.key === 'Escape') {
        label.value = object.label;
        label.blur();
      }
    });
    label.addEventListener('blur', () => void rename(object, label));

    const size = document.createElement('span');
    size.className = 'dim';
    size.textContent = `${object.size.join('×')} · ${object.facing_axis ?? 'no front'}`;
    row.append(label, size);

    row.addEventListener('click', (event) => {
      if (event.target !== label) setHighlight(highlight === object.id ? null : object.id);
    });
    dom.objects.appendChild(row);
  }
}

/** Persists one object's label, and puts the field back if the server will not take it. */
async function rename(object: SceneObject, field: HTMLInputElement): Promise<void> {
  const label = field.value.trim();
  if (!run || !label || label === object.label) {
    field.value = object.label;
    return;
  }

  try {
    const saved = await postJson<{ run: Run }>(`/api/scene-runs/${encodeURIComponent(run.id)}/labels`, {
      labels: { [object.id]: label },
    });
    run = saved.run;
    renderObjects();
    // The wording every model is sent has changed, so whatever has answered is now stale.
    await refresh();
  } catch (err) {
    field.value = object.label;
    setStatus(`Rename failed: ${(err as Error).message}`, 'err');
  }
}

function setHighlight(id: string | null): void {
  highlight = id;
  for (const tile of tiles) applyTile(tile);
  for (const row of dom.objects.querySelectorAll<HTMLElement>('.object-row')) {
    row.classList.toggle('active', row.dataset.id === highlight);
  }
}

function applyView(): void {
  dom.toggleTextured.classList.toggle('active', textured);
  dom.toggleTint.classList.toggle('active', tinted);
  dom.toggleBoxes.classList.toggle('active', showBoxes);
  for (const tile of tiles) applyTile(tile);
}

/**
 * The instruction panel against the run. Until a run has one, no model can be asked, so the
 * two buttons that would spend a call are held shut. The field is left alone while it has
 * focus: a poll landing mid-sentence must not overwrite what is being typed.
 */
function syncPrompt(): void {
  const saved = run?.prompt ?? '';
  if (document.activeElement !== dom.promptInput) dom.promptInput.value = saved;

  const typed = dom.promptInput.value.trim();
  dom.savePrompt.disabled = !run || !typed || typed === saved;
  dom.savePrompt.textContent = saved ? 'Update instruction' : 'Save instruction';
  dom.addViews.disabled = !saved || !selectedModels.length;
  dom.runDefault.disabled = !saved;
  dom.promptHint.textContent = !run
    ? 'Pick a run, or start one from a scene.'
    : saved
      ? 'Every model is sent this exact wording. Changing it marks the answers already given as stale.'
      : 'No instruction yet, so no model can be asked.';
}

// ── talking to the server ────────────────────────────────────────────────────

async function failure(response: Response): Promise<string> {
  const body = (await response.json().catch(() => null)) as { error?: string } | null;
  return body?.error ?? `HTTP ${response.status}`;
}

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await failure(response));
  return response.json() as Promise<T>;
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await failure(response));
  return response.json() as Promise<T>;
}

function stopPolling(): void {
  if (poller !== null) {
    clearInterval(poller);
    poller = null;
  }
}

function ensurePolling(): void {
  if (tiles.some((tile) => tile.status === 'running')) {
    if (poller === null) poller = window.setInterval(() => void refresh(), 2000);
  } else stopPolling();
}

/** A tile's state is a function of the answers on disk and the calls in flight, so both arrive
 *  together and every tile is recomputed from them. The manifest comes with them because the
 *  instruction and the labels are editable, so it changes under the page too. */
function applyDetail(detail: Detail): void {
  run = detail.run;
  dom.promptText.textContent = `${detail.prompt.system}\n\n${'─'.repeat(40)}\n\n${detail.prompt.user}`;
  syncPrompt();

  for (const tile of tiles) {
    if (tile.model) {
      const result = detail.results.find((entry) => entry.model === tile.model) ?? null;
      const job = detail.jobs.find((entry) => entry.model === tile.model);
      tile.result = result;
      tile.status =
        job?.status === 'running' ? 'running'
        : result ? (result.stale ? 'stale' : 'done')
        : job?.status === 'failed' ? 'failed'
        : 'idle';
      tile.error = tile.status === 'failed' ? job?.error ?? 'the call failed' : null;
    }
    renderChrome(tile);
    applyTile(tile);
  }
  ensurePolling();
}

async function refresh(): Promise<void> {
  if (!run) return;
  try {
    applyDetail(await getJson<Detail>(`/api/scene-runs/${encodeURIComponent(run.id)}`));
  } catch (err) {
    setStatus(`Refresh failed: ${(err as Error).message}`, 'err');
    stopPolling();
  }
}

function teardown(): void {
  stopPolling();
  for (const tile of tiles) disposeTile(tile);
  tiles.length = 0;
  clearSources();
  run = null;
  highlight = null;
  focus = null;
  dom.objects.textContent = '';
  dom.promptText.textContent = '';
  syncPrompt();
}

async function setRun(id: string): Promise<void> {
  teardown();
  setStatus(`Loading ${id}…`, 'busy');

  try {
    const detail = await getJson<Detail>(`/api/scene-runs/${encodeURIComponent(id)}`);
    run = detail.run;
    await buildSources(run);
    renderObjects();

    tiles.push(createTile(null));
    for (const result of detail.results) tiles.push(createTile(result.model));
    layout();
    applyDetail(detail);
    frame();

    setStatus(
      !run.prompt
        ? 'No instruction yet — name what is in the room, then write one.'
        : detail.results.length
          ? `${detail.results.length} model(s) have answered. Add more from Models.`
          : 'Nothing has answered this run yet — add a model.',
      'ok',
    );
  } catch (err) {
    setStatus(`Load failed: ${(err as Error).message}`, 'err');
  }
}

/**
 * Opens a tile for a model and makes sure it has an answer. The request is posted either way:
 * the server hands back what is already on disk unless `force` is set, so opening a tile costs
 * a call only the first time.
 */
async function addView(model: string, force = false): Promise<void> {
  if (!run) {
    setStatus('Pick a run first.', 'err');
    return;
  }
  if (!run.prompt) {
    setStatus('This run has no instruction yet — write one first.', 'err');
    return;
  }

  let tile = tiles.find((entry) => entry.model === model);
  if (!tile) {
    tile = createTile(model);
    tiles.push(tile);
    layout();
  }
  tile.status = 'running';
  renderChrome(tile);
  ensurePolling();

  try {
    await postJson(`/api/scene-runs/${encodeURIComponent(run.id)}/models`, { model, force });
    await refresh();
  } catch (err) {
    tile.status = 'failed';
    tile.error = (err as Error).message;
    renderChrome(tile);
    ensurePolling();
  }
}

async function loadRuns(): Promise<void> {
  try {
    const data = await getJson<{ runs: Listing[]; default_models: string[] }>('/api/scene-runs');
    listings = data.runs;
    defaultModels = data.default_models;

    const previous = dom.runSelect.value;
    dom.runSelect.innerHTML = '<option value="">— select run —</option>';
    for (const listing of listings) {
      const option = document.createElement('option');
      option.value = listing.id;
      const scene = listing.scene.replace(/\.glb$/i, '');
      option.textContent = listing.prompt
        ? `${scene} — ${listing.prompt} (${listing.models.length})`
        : `${scene} — no instruction yet`;
      dom.runSelect.appendChild(option);
    }
    if (listings.some((entry) => entry.id === previous)) dom.runSelect.value = previous;

    const scenes = new Set(listings.map((listing) => listing.scene)).size;
    const drafts = listings.filter((listing) => !listing.prompt).length;
    dom.runHint.textContent = listings.length
      ? `${listings.length} run(s) over ${scenes} scene(s)${drafts ? `, ${drafts} without an instruction` : ''}.`
      : 'No runs yet — start one from a scene below.';
  } catch (err) {
    dom.runHint.textContent = `Failed to list runs: ${(err as Error).message}`;
  }
}

async function loadScenes(): Promise<void> {
  try {
    const { scenes } = await getJson<{ scenes: SceneInfo[] }>('/api/scenes');
    sceneCatalogue = scenes;
    const previous = dom.sceneSelect.value;
    dom.sceneSelect.textContent = '';
    for (const scene of scenes) {
      const option = document.createElement('option');
      const name = scene.file.replace(/\.glb$/i, '');
      option.value = scene.file;
      option.disabled = !!scene.error;
      const suffix = scene.error
        ? 'unreadable'
        : scene.job?.status === 'running'
          ? scene.job.stage
          : scene.job?.status === 'failed'
            ? 'identification failed'
            : scene.identified
              ? `${scene.objects.length} named objects`
              : `${scene.objects.length} objects · needs identification`;
      option.textContent = `${name} — ${suffix}`;
      option.title = scene.error ?? scene.job?.error ?? scene.objects.map((object) => object.label).join(', ');
      dom.sceneSelect.appendChild(option);
    }
    if (scenes.some((scene) => scene.file === previous)) dom.sceneSelect.value = previous;

    const running = scenes.some((scene) => scene.job?.status === 'running');
    if (running && scenePoller === null) {
      scenePoller = window.setInterval(() => void loadScenes(), 2000);
    } else if (!running && scenePoller !== null) {
      clearInterval(scenePoller);
      scenePoller = null;
    }
  } catch (err) {
    setStatus(`Failed to list scenes: ${(err as Error).message}`, 'err');
  }
}

// ── model search ─────────────────────────────────────────────────────────────

const compactNumber = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 });

function pricePerMillion(value: number | null): string | null {
  return value === null ? null : `$${(value * 1_000_000).toFixed(value * 1_000_000 < 1 ? 2 : 1)}`;
}

function modelMeta(model: ModelInfo): string {
  const values: string[] = [];
  if (model.context_length) values.push(`${compactNumber.format(model.context_length)} ctx`);
  const input = pricePerMillion(model.pricing.prompt);
  const output = pricePerMillion(model.pricing.completion);
  if (input && output) values.push(`${input}/${output} per M`);
  if (model.structured_outputs) values.push('structured');
  else if (model.response_format) values.push('JSON mode');
  return values.join(' · ');
}

function updateModelHint(): void {
  if (selectedModels.length) {
    dom.modelHint.textContent = `${selectedModels.length} selected. Run them together or keep searching.`;
  } else if (modelResults.length) {
    dom.modelHint.textContent = 'Search by model name, provider, or exact OpenRouter ID. Use arrows and Enter to select.';
  }
  syncPrompt();
}

function renderModelChips(): void {
  dom.modelChips.textContent = '';
  for (const model of selectedModels) {
    const chip = document.createElement('span');
    chip.className = 'model-picker__chip';
    chip.title = model.id;
    const name = document.createElement('span');
    name.textContent = model.name;
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.textContent = '×';
    remove.setAttribute('aria-label', `Remove ${model.name}`);
    remove.addEventListener('click', () => {
      selectedModels = selectedModels.filter((entry) => entry.id !== model.id);
      renderModelChips();
      renderModelMenu();
      dom.modelSearch.focus();
    });
    chip.append(name, remove);
    dom.modelChips.appendChild(chip);
  }
  updateModelHint();
}

function selectModel(model: ModelInfo): void {
  if (!selectedModels.some((entry) => entry.id === model.id)) selectedModels.push(model);
  dom.modelSearch.value = '';
  activeModelIndex = -1;
  renderModelChips();
  void searchModels('');
}

function renderModelMenu(): void {
  dom.modelMenu.textContent = '';
  activeModelIndex = Math.min(activeModelIndex, modelResults.length - 1);
  const selected = new Set(selectedModels.map((model) => model.id));

  if (!modelResults.length) {
    const empty = document.createElement('p');
    empty.className = 'model-picker__empty';
    empty.textContent = 'No OpenRouter models match this search.';
    dom.modelMenu.appendChild(empty);
  }

  for (const [index, model] of modelResults.entries()) {
    const option = document.createElement('button');
    option.type = 'button';
    option.className = 'model-picker__option';
    option.id = `model-option-${index}`;
    option.dataset.index = String(index);
    option.setAttribute('role', 'option');
    option.setAttribute('aria-selected', String(selected.has(model.id)));
    option.classList.toggle('active', index === activeModelIndex);
    option.classList.toggle('selected', selected.has(model.id));

    const title = document.createElement('span');
    title.className = 'model-picker__option-title';
    const name = document.createElement('strong');
    name.textContent = model.name;
    const provider = document.createElement('span');
    provider.textContent = model.provider;
    title.append(name, provider);

    const id = document.createElement('span');
    id.className = 'model-picker__option-id';
    id.textContent = model.id;
    const meta = document.createElement('span');
    meta.className = 'model-picker__option-meta';
    meta.textContent = modelMeta(model);
    option.append(title, id, meta);
    option.title = model.description || model.id;
    option.addEventListener('pointerdown', (event) => event.preventDefault());
    option.addEventListener('click', () => selectModel(model));
    option.addEventListener('mousemove', () => {
      if (activeModelIndex !== index) {
        activeModelIndex = index;
        renderModelMenu();
      }
    });
    dom.modelMenu.appendChild(option);
  }

  dom.modelSearch.setAttribute('aria-activedescendant', activeModelIndex >= 0 ? `model-option-${activeModelIndex}` : '');
}

function openModelMenu(): void {
  dom.modelMenu.hidden = false;
  dom.modelSearch.setAttribute('aria-expanded', 'true');
}

function closeModelMenu(): void {
  dom.modelMenu.hidden = true;
  activeModelIndex = -1;
  dom.modelSearch.setAttribute('aria-expanded', 'false');
  dom.modelSearch.removeAttribute('aria-activedescendant');
}

async function searchModels(query: string): Promise<void> {
  const request = ++modelSearchRequest;
  dom.modelHint.textContent = 'Searching the OpenRouter catalogue…';
  try {
    const params = new URLSearchParams();
    if (query.trim()) params.set('q', query.trim());
    const suffix = params.size ? `?${params}` : '';
    const data = await getJson<{ models: ModelInfo[]; total: number }>(`/api/scene-models${suffix}`);
    if (request !== modelSearchRequest) return;
    modelResults = data.models;
    activeModelIndex = modelResults.length ? 0 : -1;
    renderModelMenu();
    updateModelHint();
  } catch (err) {
    if (request !== modelSearchRequest) return;
    modelResults = [];
    renderModelMenu();
    dom.modelHint.textContent = `Model catalogue unavailable: ${(err as Error).message}`;
  }
}

function queueModelSearch(): void {
  if (modelSearchTimer !== null) clearTimeout(modelSearchTimer);
  modelSearchTimer = window.setTimeout(() => {
    modelSearchTimer = null;
    void searchModels(dom.modelSearch.value);
  }, 180);
}

// ── wiring ───────────────────────────────────────────────────────────────────

dom.rescan.addEventListener('click', () => void loadRuns().then(loadScenes));
dom.frameView.addEventListener('click', () => frame());
dom.planView.addEventListener('click', () => frame(PLAN));

dom.runSelect.addEventListener('change', () => {
  const id = dom.runSelect.value;
  // Keyboard shortcuts skip a focused select, so hand focus back to the page.
  dom.runSelect.blur();
  if (id) void setRun(id);
  else {
    teardown();
    setStatus('');
  }
});

dom.togglePrompt.addEventListener('click', () => {
  dom.promptText.hidden = !dom.promptText.hidden;
  dom.togglePrompt.classList.toggle('active', !dom.promptText.hidden);
});

dom.modelControl.addEventListener('click', () => dom.modelSearch.focus());

dom.modelSearch.addEventListener('focus', () => {
  openModelMenu();
  if (!modelResults.length) void searchModels(dom.modelSearch.value);
});

dom.modelSearch.addEventListener('input', () => {
  openModelMenu();
  queueModelSearch();
});

dom.modelSearch.addEventListener('keydown', (event) => {
  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
    event.preventDefault();
    openModelMenu();
    if (!modelResults.length) return;
    const direction = event.key === 'ArrowDown' ? 1 : -1;
    activeModelIndex = (activeModelIndex + direction + modelResults.length) % modelResults.length;
    renderModelMenu();
    dom.modelMenu.querySelector<HTMLElement>(`[data-index="${activeModelIndex}"]`)?.scrollIntoView({ block: 'nearest' });
  } else if (event.key === 'Enter' && !dom.modelMenu.hidden && activeModelIndex >= 0) {
    event.preventDefault();
    selectModel(modelResults[activeModelIndex]);
  } else if (event.key === 'Escape') {
    event.preventDefault();
    closeModelMenu();
  } else if (event.key === 'Backspace' && !dom.modelSearch.value && selectedModels.length) {
    selectedModels.pop();
    renderModelChips();
    renderModelMenu();
  }
});

document.addEventListener('pointerdown', (event) => {
  if (!dom.modelPicker.contains(event.target as Node)) closeModelMenu();
});

dom.addViews.addEventListener('click', () => {
  const models = selectedModels.map((model) => model.id);
  if (!models.length) return;
  selectedModels = [];
  renderModelChips();
  closeModelMenu();
  for (const model of models) void addView(model);
});

dom.runDefault.addEventListener('click', () => {
  for (const model of defaultModels) void addView(model);
});

dom.promptInput.addEventListener('input', syncPrompt);

dom.savePrompt.addEventListener('click', async () => {
  if (!run) return;
  setStatus('Saving the instruction…', 'busy');
  try {
    const saved = await postJson<{ run: Run }>(`/api/scene-runs/${encodeURIComponent(run.id)}/prompt`, {
      prompt: dom.promptInput.value.trim(),
    });
    run = saved.run;
    dom.promptInput.blur();
    syncPrompt();
    // The dropdown labels runs by their instruction, and every answer already given is stale.
    await loadRuns();
    await refresh();
    setStatus('Instruction saved.', 'ok');
  } catch (err) {
    setStatus(`Save failed: ${(err as Error).message}`, 'err');
  }
});

dom.toggleTextured.addEventListener('click', () => {
  textured = !textured;
  applyView();
});

dom.toggleTint.addEventListener('click', () => {
  tinted = !tinted;
  applyView();
});

dom.toggleBoxes.addEventListener('click', () => {
  showBoxes = !showBoxes;
  applyView();
});

dom.columns.addEventListener('change', () => {
  columns = Number(dom.columns.value);
  dom.columns.blur();
  layout();
});

dom.createRun.addEventListener('click', async () => {
  const scene = dom.sceneSelect.value;
  if (!scene) {
    setStatus('Pick a scene to start from.', 'err');
    return;
  }

  const info = sceneCatalogue.find((entry) => entry.file === scene);
  if (!info?.identified) {
    if (info?.job?.status === 'running') {
      setStatus(`Identifying ${scene}: ${info.job.stage}`, 'busy');
      return;
    }
    setStatus(`Rendering four views and identifying every object in ${scene}…`, 'busy');
    try {
      await postJson(`/api/scenes/${encodeURIComponent(scene)}/identify`, {});
      await loadScenes();
      setStatus('Identification started. Create the run once the scene is marked named.', 'busy');
    } catch (err) {
      setStatus(`Identification failed to start: ${(err as Error).message}`, 'err');
    }
    return;
  }

  setStatus('Creating the run…', 'busy');
  try {
    const { run: created } = await postJson<{ run: Run }>('/api/scene-runs', { scene });
    await loadRuns();
    dom.runSelect.value = created.id;
    await setRun(created.id);
  } catch (err) {
    setStatus(`Create failed: ${(err as Error).message}`, 'err');
  }
});

let pressed: { x: number; y: number } | null = null;

renderer.domElement.addEventListener('pointerdown', (event) => {
  pressed = { x: event.clientX, y: event.clientY };
});

renderer.domElement.addEventListener('pointerup', (event) => {
  const from = pressed;
  pressed = null;
  // An orbit drag ends in a pointerup too, so only a press that barely moved is a click.
  if (!from || Math.hypot(event.clientX - from.x, event.clientY - from.y) > 4) return;
  setHighlight(objectAt(event.clientX, event.clientY));
});

/** Which object is under the pointer, in whichever tile the pointer is over. */
function objectAt(clientX: number, clientY: number): string | null {
  const tile = tiles.find((entry) => {
    if (entry.root.hidden) return false;
    const rect = entry.view.getBoundingClientRect();
    return clientX >= rect.left && clientX <= rect.right && clientY >= rect.top && clientY <= rect.bottom;
  });
  if (!tile) return null;

  const rect = tile.view.getBoundingClientRect();
  pointer.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
  camera.aspect = rect.width / rect.height;
  camera.updateProjectionMatrix();
  raycaster.setFromCamera(pointer, camera);

  const roots = [...tile.objects.values()].filter((object) => object.pivot.visible).map((object) => object.pivot);
  let node: THREE.Object3D | null = raycaster.intersectObjects(roots, true)[0]?.object ?? null;
  while (node && node.userData.objectId === undefined) node = node.parent;
  return (node?.userData.objectId as string) ?? null;
}

window.addEventListener('keydown', (event) => {
  const target = event.target as HTMLElement;
  if (target?.tagName === 'INPUT' || target?.tagName === 'TEXTAREA' || target?.tagName === 'SELECT') return;
  const key = event.key.toLowerCase();
  if (key === 'f') frame();
  else if (key === 't') frame(PLAN);
  else if (event.key === 'Escape') {
    focus = null;
    layout();
    setHighlight(null);
  }
});

applyView();
syncPrompt();
renderModelChips();
void searchModels('');
void loadScenes();
void loadRuns().then(() => {
  const id = new URLSearchParams(location.search).get('run') ?? listings[0]?.id;
  if (id && listings.some((listing) => listing.id === id)) {
    dom.runSelect.value = id;
    void setRun(id);
  }
});
