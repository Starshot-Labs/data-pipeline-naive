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

const LAYERS: Layer[] = ['anchor', 'placed'];
const TAGS: Record<Layer, string> = { anchor: 'tag-a', placed: 'tag-b' };

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
  frameView: el<HTMLButtonElement>('frameView'),
  status: el<HTMLParagraphElement>('status'),
  selInfo: el<HTMLDivElement>('selInfo'),
  placement: el<HTMLParagraphElement>('placement'),
  numbers: el<HTMLDivElement>('numbers'),
  descriptions: el<HTMLDivElement>('descriptions'),
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

const layers: Partial<Record<Layer, THREE.Object3D>> = {};
const visible: Record<Layer, boolean> = { anchor: true, placed: true };
let runs: Run[] = [];

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

function clearLayers(): void {
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
  const base = `/generated/${encodeURIComponent(id)}`;

  try {
    // Served out of the sample folder when its meshes are there, off the scene volume when they
    // are not. Either way, whatever pose a mesh has is already inside the file.
    const loaded = await Promise.all(
      run.meshes.map((mesh) => loader.loadAsync(`/mesh/${encodeURIComponent(id)}/${encodeURIComponent(mesh)}`)),
    );
    for (const [i, layer] of LAYERS.entries()) {
      layers[layer] = loaded[i].scene;
      scene.add(loaded[i].scene);
    }
    applyVisibility();

    if (run.placed) {
      const response = await fetch(`${base}/metadata.json`);
      if (!response.ok) throw new Error(`metadata.json: HTTP ${response.status}`);
      renderInfo((await response.json()) as Sample, base);
    } else {
      renderFiles(run);
    }

    frame();
    setStatus(
      run.placed
        ? 'Loaded posed GLBs — the arrangement comes only from the files.'
        : 'Loaded raw GLBs — this folder has no metadata.json, so neither mesh is posed.',
      'ok',
    );
  } catch (err) {
    setStatus(`Load failed: ${(err as Error).message}`, 'err');
  }
}

async function loadRuns(): Promise<void> {
  try {
    const response = await fetch('/api/runs');
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

for (const key of LAYERS) {
  buttons[key].addEventListener('click', () => {
    visible[key] = !visible[key];
    applyVisibility();
  });
}

dom.reloadRuns.addEventListener('click', () => void loadRuns());
dom.frameView.addEventListener('click', frame);

dom.runSelect.addEventListener('change', () => {
  const id = dom.runSelect.value;
  if (id) void loadRun(id);
  else {
    clearLayers();
    applyVisibility();
    setStatus('');
  }
});

window.addEventListener('keydown', (event) => {
  const target = event.target as HTMLElement;
  if (target?.tagName === 'INPUT' || target?.tagName === 'TEXTAREA' || target?.tagName === 'SELECT') return;
  if (event.key.toLowerCase() === 'f') frame();
});

applyVisibility();
void loadRuns().then(() => {
  const id = new URLSearchParams(location.search).get('run') ?? runs[0]?.id;
  if (id && runs.some((run) => run.id === id)) {
    dom.runSelect.value = id;
    void loadRun(id);
  }
});
