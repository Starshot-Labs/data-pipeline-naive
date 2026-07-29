import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { addStudioLighting } from './viewer';

type Vec3 = [number, number, number];
type Layer = 'input' | 'mask' | 'result';

/** One entry from `/api/edits`. `edited` is false while only the local half has run. */
interface Listing {
  id: string;
  placement: string;
  edited: boolean;
  views: string[];
}

/** A sample's mask.json — see pipeline/test-edit.mjs. */
interface Mask {
  id: string;
  placement: string;
  model: string;
  anchor: {
    source: string;
    mesh: string;
    normalization: { position: Vec3; scale: number };
    solid_cells: number;
  };
  object: { source_image: string; name: string };
  mask: {
    mesh: string;
    grid: number;
    box: { min: Vec3; max: Vec3 };
    cells: number;
    triangles: number;
    volume: number;
    reasoning: string;
  };
  preview: string;
  usage?: { cost?: number } | null;
}

/** A sample's edit.json, written only once the service has returned. */
interface Edit {
  job_id?: string;
  view?: string;
  total_seconds?: number;
  /** Missing on runs made before it was recorded, hence the fallbacks where it is read. */
  result?: { mesh: string; preview: string; triangles: number };
}

const LAYERS: Layer[] = ['input', 'mask', 'result'];
/** Half-extent of the cube every file in edit-results is authored in. */
const HALF = 0.5;

// The service's order, which is the order the 2D pass runs in.
const VIEWS: [string, string][] = [
  ['2d_render.png', 'the view it picked'],
  ['2d_mask.png', "that view's mask, by depth"],
  ['2d_edit.png', 'FLUX inpaint of the hole'],
];

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
  showAll: el<HTMLButtonElement>('showAll'),
  toggleGhost: el<HTMLButtonElement>('toggleGhost'),
  toggleTextured: el<HTMLButtonElement>('toggleTextured'),
  toggleBox: el<HTMLButtonElement>('toggleBox'),
  frameView: el<HTMLButtonElement>('frameView'),
  status: el<HTMLParagraphElement>('status'),
  selInfo: el<HTMLDivElement>('selInfo'),
  placement: el<HTMLParagraphElement>('placement'),
  numbers: el<HTMLDivElement>('numbers'),
  reasoning: el<HTMLParagraphElement>('reasoning'),
  cards: el<HTMLDivElement>('cards'),
};

const toggles: Record<Layer, HTMLButtonElement> = {
  input: el<HTMLButtonElement>('toggleInput'),
  mask: el<HTMLButtonElement>('toggleMask'),
  result: el<HTMLButtonElement>('toggleResult'),
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

// Everything here is authored in TRELLIS's cube — longest axis 1, bounding box centred on the
// origin — and nothing rests on y=0, so the cube is the reference worth drawing rather than
// the ground plane the other viewers use.
const cube = new THREE.Box3Helper(
  new THREE.Box3(new THREE.Vector3(-HALF, -HALF, -HALF), new THREE.Vector3(HALF, HALF, HALF)),
  0x2a313c,
);
scene.add(cube, new THREE.AxesHelper(HALF));

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 5000);
camera.position.set(1.2, 0.9, 1.6);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;

const loader = new GLTFLoader();

/** Shared, so switching samples only has to dispose what came out of a GLB. */
const GHOST = new THREE.MeshStandardMaterial({
  color: 0x8b949e,
  roughness: 0.7,
  transparent: true,
  opacity: 0.16,
  depthWrite: false,
  side: THREE.DoubleSide,
});

const FLAT: Record<Layer, THREE.Material> = {
  input: new THREE.MeshStandardMaterial({ color: 0x8b949e, roughness: 0.6 }),
  // The mask is a hollow shell you often end up inside of, and carries no texture either way.
  mask: new THREE.MeshStandardMaterial({ color: 0xf85149, roughness: 0.6, side: THREE.DoubleSide }),
  result: new THREE.MeshStandardMaterial({ color: 0x5ec8ff, roughness: 0.6 }),
};

let listings: Listing[] = [];
let current: Listing | null = null;
const layers: Partial<Record<Layer, THREE.Object3D>> = {};
const visible: Record<Layer, boolean> = { input: true, mask: true, result: true };
let boxHelper: THREE.Box3Helper | null = null;
let ghost = true;
let textured = false;
let showBox = false;

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

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url.slice(url.lastIndexOf('/') + 1)}: HTTP ${response.status}`);
  return response.json() as Promise<T>;
}

/** The materials this viewer swaps in are shared, so only a GLB's own are disposed. */
function dispose(root: THREE.Object3D): void {
  scene.remove(root);
  root.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (mesh.geometry) mesh.geometry.dispose();
    const original = mesh.userData.original as THREE.Material | undefined;
    if (!original) return;
    for (const value of Object.values(original)) if (value instanceof THREE.Texture) value.dispose();
    original.dispose();
  });
}

function clearScene(): void {
  for (const key of LAYERS) {
    const root = layers[key];
    if (root) dispose(root);
    delete layers[key];
  }
  if (boxHelper) {
    scene.remove(boxHelper);
    boxHelper.geometry.dispose();
    boxHelper = null;
  }
}

/**
 * `model.glb` and `output.glb` fill the same cube, and a mask for "inside" sits behind the
 * anchor's near wall, so the input has to be able to go see-through — the same reason
 * mask-view.png dithers the anchor rather than drawing it solid.
 */
function paint(layer: Layer, root: THREE.Object3D): void {
  root.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (!mesh.isMesh) return;
    const original = (mesh.userData.original ??= mesh.material as THREE.Material) as THREE.Material;
    if (layer === 'input' && ghost) mesh.material = GHOST;
    else if (layer === 'mask' || !textured) mesh.material = FLAT[layer];
    else mesh.material = original;
  });
}

function applyView(): void {
  for (const key of LAYERS) {
    const root = layers[key];
    if (root) {
      root.visible = visible[key];
      paint(key, root);
    }
    toggles[key].classList.toggle('active', !!root && visible[key]);
    toggles[key].disabled = !root;
  }

  if (boxHelper) boxHelper.visible = showBox;
  dom.toggleBox.disabled = !boxHelper;
  dom.toggleGhost.classList.toggle('active', ghost);
  dom.toggleTextured.classList.toggle('active', textured);
  dom.toggleBox.classList.toggle('active', showBox);

  const shown = LAYERS.filter((key) => layers[key] && visible[key]);
  dom.selInfo.textContent = current
    ? `${current.id} · "${current.placement}" · ${shown.join(' + ') || 'nothing shown'}`
    : 'Pick a sample';
}

function frame(): void {
  const bounds = new THREE.Box3();
  for (const key of LAYERS) {
    const root = layers[key];
    if (root && visible[key]) bounds.expandByObject(root);
  }
  if (bounds.isEmpty()) bounds.copy(cube.box);

  const sphere = bounds.getBoundingSphere(new THREE.Sphere());
  const direction = new THREE.Vector3(0.6, 0.45, 1).normalize();
  const distance = (sphere.radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.45;
  orbit.target.copy(sphere.center);
  camera.position.copy(sphere.center).addScaledVector(direction, Math.max(distance, 0.4));
  camera.near = Math.max(distance / 100, 0.001);
  camera.far = distance * 100;
  camera.updateProjectionMatrix();
  orbit.update();
}

function showEverything(): void {
  for (const key of LAYERS) visible[key] = true;
  applyView();
}

/** The LLM's answer as it was given: inclusive cell indices on the lattice mask.mjs carves on. */
function gridBox({ grid, box }: Mask['mask']): THREE.Box3Helper {
  const corner = (cell: Vec3, offset: number) =>
    new THREE.Vector3(cell[0] + offset, cell[1] + offset, cell[2] + offset).divideScalar(grid).subScalar(HALF);
  return new THREE.Box3Helper(new THREE.Box3(corner(box.min, 0), corner(box.max, 1)), 0xffd166);
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

function card(title: string): HTMLElement {
  const section = document.createElement('section');
  section.className = 'panel';
  const head = document.createElement('div');
  head.className = 'panel-title';
  const label = document.createElement('span');
  label.textContent = title;
  head.appendChild(label);
  section.appendChild(head);
  dom.cards.appendChild(section);
  return section;
}

function renderInfo(base: string, meta: Mask, edit: Edit | null, views: string[]): void {
  dom.placement.textContent = `"${meta.placement}"`;
  dom.reasoning.textContent = meta.mask.reasoning;

  const { grid, box, cells, triangles, volume } = meta.mask;
  const span = [0, 1, 2].map((axis) => box.max[axis] - box.min[axis] + 1);
  const rows: [string, string][] = [
    ['object', meta.object.name],
    ['anchor', meta.anchor.source],
    ['box', `[${box.min.join(', ')}] → [${box.max.join(', ')}]`],
    ['box cells', `${span.join(' × ')} = ${span[0] * span[1] * span[2]}`],
    ['mask cells', `${cells}, ${(volume * 100).toFixed(2)}% of cube`],
    ['mask tris', String(triangles)],
    ['anchor cells', `${meta.anchor.solid_cells} of ${grid ** 3}`],
    ['model', meta.model],
  ];
  if (meta.usage?.cost) rows.push(['cost', `$${meta.usage.cost.toFixed(4)}`]);
  if (edit) {
    if (edit.job_id) rows.push(['job', edit.job_id]);
    if (edit.view) rows.push(['view', edit.view]);
    if (edit.result) rows.push(['result tris', String(edit.result.triangles)]);
    if (edit.total_seconds) rows.push(['runtime', `${(edit.total_seconds / 60).toFixed(1)} min`]);
  }

  dom.numbers.textContent = '';
  for (const [key, value] of rows) row(dom.numbers, key, value);

  dom.cards.textContent = '';
  const previews = card('Previews');
  image(previews, `object · ${meta.object.source_image}`, `${base}/source.png`);
  image(previews, 'mask region, anchor ghosted', `${base}/${meta.preview}`);
  if (edit) image(previews, 'edit result', `${base}/${edit.result?.preview ?? 'result-view.png'}`);

  const produced = VIEWS.filter(([file]) => views.includes(file));
  if (produced.length) {
    const pass = card('VoxHammer 2D pass');
    for (const [file, caption] of produced) image(pass, caption, `${base}/images/${file}`);
  }
}

async function loadSample(id: string): Promise<void> {
  const listing = listings.find((entry) => entry.id === id);
  if (!listing) return;

  clearScene();
  current = listing;
  setStatus(`Loading ${id}…`, 'busy');
  const base = `/edit-results/${encodeURIComponent(id)}`;

  try {
    const meta = await getJson<Mask>(`${base}/mask.json`);
    const edit = listing.edited ? await getJson<Edit>(`${base}/edit.json`) : null;

    const meshes: Partial<Record<Layer, string>> = {
      input: meta.anchor.mesh,
      mask: meta.mask.mesh,
      result: edit ? edit.result?.mesh ?? 'output.glb' : undefined,
    };
    const present = LAYERS.filter((key) => meshes[key]);
    const gltfs = await Promise.all(present.map((key) => loader.loadAsync(`${base}/${meshes[key]}`)));
    for (const [index, key] of present.entries()) {
      layers[key] = gltfs[index].scene;
      scene.add(gltfs[index].scene);
    }

    boxHelper = gridBox(meta.mask);
    scene.add(boxHelper);

    // A finished sample opens on what it produced, an unfinished one on the region it will fill.
    visible.input = !edit;
    visible.mask = !edit;
    visible.result = !!edit;

    renderInfo(base, meta, edit, listing.views);
    applyView();
    frame();
    setStatus(
      edit
        ? 'Edited — press 1 for the input it grew out of.'
        : 'Mask only — the GPU half has not run this sample.',
      'ok',
    );
  } catch (err) {
    setStatus(`Load failed: ${(err as Error).message}`, 'err');
  }
}

async function loadListings(): Promise<void> {
  try {
    listings = (await getJson<{ samples: Listing[] }>('/api/edits')).samples ?? [];

    const previous = dom.runSelect.value;
    dom.runSelect.innerHTML = '<option value="">— select sample —</option>';
    for (const entry of listings) {
      const option = document.createElement('option');
      option.value = entry.id;
      option.textContent = `${entry.id} — ${entry.placement}${entry.edited ? '' : ' (mask only)'}`;
      dom.runSelect.appendChild(option);
    }
    if (listings.some((entry) => entry.id === previous)) dom.runSelect.value = previous;

    const edited = listings.filter((entry) => entry.edited).length;
    dom.runHint.textContent = listings.length
      ? `${listings.length} sample(s), ${edited} edited.`
      : 'Nothing yet — run `npm run test-edit`.';
  } catch (err) {
    dom.runHint.textContent = `Failed to list: ${(err as Error).message}`;
  }
}

for (const key of LAYERS) {
  toggles[key].addEventListener('click', () => {
    visible[key] = !visible[key];
    applyView();
  });
}

dom.reloadRuns.addEventListener('click', () => void loadListings());
dom.frameView.addEventListener('click', frame);
dom.showAll.addEventListener('click', showEverything);

dom.toggleGhost.addEventListener('click', () => {
  ghost = !ghost;
  applyView();
});

dom.toggleTextured.addEventListener('click', () => {
  textured = !textured;
  applyView();
});

dom.toggleBox.addEventListener('click', () => {
  showBox = !showBox;
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
    dom.numbers.textContent = '';
    dom.cards.textContent = '';
    dom.placement.textContent = '';
    dom.reasoning.textContent = '';
    applyView();
    setStatus('');
  }
});

window.addEventListener('keydown', (event) => {
  const target = event.target as HTMLElement;
  if (target?.tagName === 'INPUT' || target?.tagName === 'TEXTAREA' || target?.tagName === 'SELECT') return;
  const key = event.key.toLowerCase();
  if (key === 'f') frame();
  else if (key === 'g') {
    ghost = !ghost;
    applyView();
  } else if (event.key === '0') showEverything();
  else if (/^[1-3]$/.test(event.key)) {
    const layer = LAYERS[Number(event.key) - 1];
    if (layers[layer]) {
      visible[layer] = !visible[layer];
      applyView();
    }
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
