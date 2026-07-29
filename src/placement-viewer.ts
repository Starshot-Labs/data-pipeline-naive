import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { addStudioLighting } from './viewer';

type Vec3 = [number, number, number];
type Quat = [number, number, number, number];

interface ModelEntry {
  slug: string;
  model: string;
}

/** One entry from `/api/placements`. `models` is empty until something places it. */
interface Listing {
  id: string;
  placement: string;
  anchor_view: string | null;
  generated_image: string | null;
  models: ModelEntry[];
}

/** A model's placement.json — see pipeline/test-placement.mjs. */
interface Placement {
  id: string;
  model: string;
  model_used?: string;
  latency_ms?: number;
  answer: {
    reasoning: string;
    scale: number;
    yaw_degrees: number;
    pitch_degrees: number;
    roll_degrees: number;
    center: Vec3;
  };
  transform: { position: Vec3; rotation: Quat; scale: Vec3 };
  scene_image?: string;
  usage?: { cost?: number; total_tokens?: number } | null;
}

interface Loaded extends ModelEntry {
  object: THREE.Object3D;
  meta: Placement;
  color: number;
  visible: boolean;
}

// Enough to tell four to six overlaid poses apart on a dark background.
const PALETTE = [0xff8f5e, 0x5ec8ff, 0x3fb950, 0xd2a8ff, 0xffd166, 0xf85149];
const hex = (color: number) => `#${color.toString(16).padStart(6, '0')}`;

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node as T;
}

const dom = {
  viewport: el<HTMLDivElement>('viewport'),
  runSelect: el<HTMLSelectElement>('runSelect'),
  reloadRuns: el<HTMLButtonElement>('reloadRuns'),
  runHint: el<HTMLParagraphElement>('runHint'),
  models: el<HTMLDivElement>('models'),
  showAll: el<HTMLButtonElement>('showAll'),
  toggleAnchor: el<HTMLButtonElement>('toggleAnchor'),
  toggleTextured: el<HTMLButtonElement>('toggleTextured'),
  frameView: el<HTMLButtonElement>('frameView'),
  status: el<HTMLParagraphElement>('status'),
  selInfo: el<HTMLDivElement>('selInfo'),
  placement: el<HTMLParagraphElement>('placement'),
  shared: el<HTMLDivElement>('shared'),
  cards: el<HTMLDivElement>('cards'),
};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.outputColorSpace = THREE.SRGBColorSpace;
dom.viewport.appendChild(renderer.domElement);

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
addStudioLighting(scene);

const grid = new THREE.GridHelper(20, 20, 0x2a313c, 0x1c2230);
(grid.material as THREE.Material).transparent = true;
(grid.material as THREE.Material).opacity = 0.55;
scene.add(grid, new THREE.AxesHelper(0.5));

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 5000);
camera.position.set(2.5, 1.8, 3.2);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;

const loader = new GLTFLoader();

let listings: Listing[] = [];
/** Colour per model id, indexed off the whole benchmark so it never shifts per sample. */
const colors = new Map<string, number>();
let current: Listing | null = null;
let anchor: THREE.Object3D | null = null;
let loaded: Loaded[] = [];
let showAnchor = true;
let textured = false;

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

function dispose(root: THREE.Object3D): void {
  scene.remove(root);
  root.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (mesh.geometry) mesh.geometry.dispose();
    const tint = mesh.userData?.tint as THREE.Material | undefined;
    if (tint) tint.dispose();
  });
}

function clearScene(): void {
  if (anchor) dispose(anchor);
  for (const entry of loaded) dispose(entry.object);
  anchor = null;
  loaded = [];
}

/**
 * Every model's mesh is the same object in a different pose, so they are flat-shaded
 * in the model's colour — overlapping textures would be unreadable. `Textured` puts
 * the original materials back, which is what you want when soloing one model.
 */
function paint(root: THREE.Object3D, color: number): void {
  root.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (!mesh.isMesh) return;
    mesh.userData.original ??= mesh.material;
    mesh.userData.tint ??= new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.05 });
    mesh.material = textured ? mesh.userData.original : mesh.userData.tint;
  });
}

function applyView(): void {
  if (anchor) anchor.visible = showAnchor;
  for (const entry of loaded) {
    entry.object.visible = entry.visible;
    paint(entry.object, entry.color);
  }
  dom.toggleAnchor.classList.toggle('active', showAnchor);
  dom.toggleTextured.classList.toggle('active', textured);

  for (const button of dom.models.querySelectorAll<HTMLButtonElement>('button[data-slug]')) {
    const entry = loaded.find((item) => item.slug === button.dataset.slug);
    button.classList.toggle('active', !!entry?.visible);
  }

  const shown = loaded.filter((entry) => entry.visible).length;
  dom.selInfo.textContent = current
    ? `${current.id} · "${current.placement}" · ${shown}/${loaded.length} model(s)`
    : 'Pick a sample';
}

function frame(): void {
  const box = new THREE.Box3();
  if (anchor?.visible) box.expandByObject(anchor);
  for (const entry of loaded) if (entry.visible) box.expandByObject(entry.object);
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

function solo(index: number): void {
  for (const [i, entry] of loaded.entries()) entry.visible = i === index;
  applyView();
}

function row(parent: HTMLElement, key: string, value: string): void {
  const line = document.createElement('div');
  line.className = 'bbox-row';
  const label = document.createElement('span');
  label.textContent = key;
  const text = document.createElement('span');
  text.textContent = value;
  line.append(label, text);
  parent.appendChild(line);
}

function image(parent: HTMLElement, caption: string, src: string): void {
  const card = document.createElement('div');
  card.className = 'part';
  const label = document.createElement('p');
  label.className = 'hint';
  label.textContent = caption;
  const img = document.createElement('img');
  img.src = src;
  img.alt = caption;
  img.loading = 'lazy';
  card.append(label, img);
  parent.appendChild(card);
}

function renderToggles(): void {
  dom.models.textContent = '';
  if (!loaded.length) {
    const empty = document.createElement('p');
    empty.className = 'hint';
    empty.textContent = 'No model has placed this sample yet.';
    dom.models.appendChild(empty);
    return;
  }

  for (const [index, entry] of loaded.entries()) {
    const button = document.createElement('button');
    button.className = 'model';
    button.dataset.slug = entry.slug;
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = hex(entry.color);
    const label = document.createElement('span');
    label.textContent = entry.model;
    button.append(swatch, label);
    button.title = `${entry.model} — click to toggle, press ${index + 1} to solo`;
    button.addEventListener('click', () => {
      entry.visible = !entry.visible;
      applyView();
    });
    dom.models.appendChild(button);
  }
}

function renderCards(base: string): void {
  dom.cards.textContent = '';
  for (const entry of loaded) {
    const { answer, transform } = entry.meta;
    const meta = entry.meta;
    const panel = document.createElement('section');
    panel.className = 'panel';

    const title = document.createElement('div');
    title.className = 'panel-title';
    const name = document.createElement('span');
    name.className = 'model-name';
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = hex(entry.color);
    name.append(swatch, entry.model);
    title.appendChild(name);
    panel.appendChild(title);

    const numbers = document.createElement('div');
    numbers.className = 'bbox';
    row(numbers, 'scale', String(answer.scale));
    row(numbers, 'yaw / pitch / roll', `${answer.yaw_degrees}° / ${answer.pitch_degrees}° / ${answer.roll_degrees}°`);
    row(numbers, 'center', answer.center.map((v) => v.toFixed(2)).join(', '));
    row(numbers, 'position', transform.position.map((v) => v.toFixed(4)).join(', '));
    if (meta.latency_ms) row(numbers, 'latency', `${(meta.latency_ms / 1000).toFixed(1)} s`);
    if (meta.usage?.cost) row(numbers, 'cost', `$${meta.usage.cost.toFixed(4)}`);
    panel.appendChild(numbers);

    if (meta.scene_image) image(panel, 'scene', `${base}/${entry.slug}/${meta.scene_image}`);

    const reasoning = document.createElement('p');
    reasoning.className = 'placement';
    reasoning.style.color = 'var(--muted)';
    reasoning.textContent = answer.reasoning;
    panel.appendChild(reasoning);

    dom.cards.appendChild(panel);
  }
}

async function loadSample(id: string): Promise<void> {
  const listing = listings.find((entry) => entry.id === id);
  if (!listing) return;

  clearScene();
  current = listing;
  setStatus(`Loading ${id}…`, 'busy');
  const base = `/placement-results/${encodeURIComponent(id)}`;

  dom.placement.textContent = `"${listing.placement}"`;
  dom.shared.textContent = '';
  if (listing.anchor_view) image(dom.shared, 'anchor view', `${base}/${listing.anchor_view}`);
  if (listing.generated_image) image(dom.shared, 'generated object', `${base}/${listing.generated_image}`);

  try {
    const [anchorGltf, ...models] = await Promise.all([
      loader.loadAsync(`${base}/anchor.glb`),
      ...listing.models.map(async (entry) => {
        const folder = `${base}/${entry.slug}`;
        const [gltf, meta] = await Promise.all([
          loader.loadAsync(`${folder}/placed.glb`),
          fetch(`${folder}/placement.json`).then(async (response) => {
            if (!response.ok) throw new Error(`${entry.slug}/placement.json: HTTP ${response.status}`);
            return response.json() as Promise<Placement>;
          }),
        ]);
        return { ...entry, object: gltf.scene, meta };
      }),
    ]);

    anchor = anchorGltf.scene;
    scene.add(anchor);
    loaded = models.map((entry) => ({
      ...entry,
      color: colors.get(entry.model) ?? PALETTE[0],
      visible: true,
    }));
    for (const entry of loaded) scene.add(entry.object);

    renderToggles();
    renderCards(base);
    applyView();
    frame();
    setStatus(
      loaded.length
        ? `${loaded.length} model(s) loaded — each pose comes only from its own file.`
        : 'Meshes loaded — no model results yet, run `npm run test-place`.',
      'ok',
    );
  } catch (err) {
    setStatus(`Load failed: ${(err as Error).message}`, 'err');
  }
}

async function loadListings(): Promise<void> {
  try {
    const response = await fetch('/api/placements');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    listings = ((await response.json()) as { samples: Listing[] }).samples ?? [];

    // Colours are assigned over every model in the benchmark, sorted, so one model
    // keeps its colour as you move between samples.
    const every = [...new Set(listings.flatMap((entry) => entry.models.map((model) => model.model)))].sort();
    colors.clear();
    for (const [index, model] of every.entries()) colors.set(model, PALETTE[index % PALETTE.length]);

    const previous = dom.runSelect.value;
    dom.runSelect.innerHTML = '<option value="">— select sample —</option>';
    for (const entry of listings) {
      const option = document.createElement('option');
      option.value = entry.id;
      option.textContent = `${entry.id} — ${entry.placement} (${entry.models.length})`;
      dom.runSelect.appendChild(option);
    }
    if (listings.some((entry) => entry.id === previous)) dom.runSelect.value = previous;

    dom.runHint.textContent = listings.length
      ? `${listings.length} sample(s), ${every.length} model(s) under test.`
      : 'Nothing yet — run `npm run test-placement`.';
  } catch (err) {
    dom.runHint.textContent = `Failed to list: ${(err as Error).message}`;
  }
}

dom.reloadRuns.addEventListener('click', () => void loadListings());
dom.frameView.addEventListener('click', frame);

dom.showAll.addEventListener('click', () => {
  for (const entry of loaded) entry.visible = true;
  applyView();
});

dom.toggleAnchor.addEventListener('click', () => {
  showAnchor = !showAnchor;
  applyView();
});

dom.toggleTextured.addEventListener('click', () => {
  textured = !textured;
  applyView();
});

dom.runSelect.addEventListener('change', () => {
  const id = dom.runSelect.value;
  // Keyboard shortcuts skip a focused select, so hand focus back to the page —
  // otherwise nothing responds until you click the viewport.
  dom.runSelect.blur();
  if (id) void loadSample(id);
  else {
    clearScene();
    current = null;
    renderToggles();
    dom.cards.textContent = '';
    applyView();
    setStatus('');
  }
});

window.addEventListener('keydown', (event) => {
  const target = event.target as HTMLElement;
  if (target?.tagName === 'INPUT' || target?.tagName === 'TEXTAREA' || target?.tagName === 'SELECT') return;
  if (event.key.toLowerCase() === 'f') frame();
  else if (event.key === '0') {
    for (const entry of loaded) entry.visible = true;
    applyView();
  } else if (/^[1-9]$/.test(event.key)) {
    const index = Number(event.key) - 1;
    if (index < loaded.length) solo(index);
  }
});

applyView();
void loadListings().then(() => {
  const id = new URLSearchParams(location.search).get('sample') ?? listings[0]?.id;
  if (id && listings.some((entry) => entry.id === id)) {
    dom.runSelect.value = id;
    void loadSample(id);
  }
});
