import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { addStudioLighting } from './viewer';

interface SampleSummary {
  id: string;
  glbs: string[];
}

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node as T;
}

const dom = {
  viewport: el<HTMLDivElement>('viewport'),
  sampleSelect: el<HTMLSelectElement>('sampleSelect'),
  reloadSamples: el<HTMLButtonElement>('reloadSamples'),
  sampleHint: el<HTMLParagraphElement>('sampleHint'),
  fileInfo: el<HTMLParagraphElement>('fileInfo'),
  toggleA: el<HTMLButtonElement>('toggleA'),
  toggleB: el<HTMLButtonElement>('toggleB'),
  frameView: el<HTMLButtonElement>('frameView'),
  status: el<HTMLParagraphElement>('status'),
  selInfo: el<HTMLDivElement>('selInfo'),
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
scene.add(grid);
scene.add(new THREE.AxesHelper(0.5));

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 5000);
camera.position.set(2.5, 1.8, 3.2);

const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;

const loader = new GLTFLoader();
let samples: SampleSummary[] = [];
// roots[0] = anchor GLB ("A"), roots[1..] = the rest ("B", ...) — loaded RAW.
let roots: THREE.Object3D[] = [];

function resize(): void {
  const { clientWidth: w, clientHeight: h } = dom.viewport;
  if (!w || !h) return;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
new ResizeObserver(resize).observe(dom.viewport);
resize();

function animate(): void {
  requestAnimationFrame(animate);
  orbit.update();
  renderer.render(scene, camera);
}
animate();

function setStatus(text: string, kind: '' | 'ok' | 'err' | 'busy' = ''): void {
  dom.status.textContent = text;
  dom.status.className = `status ${kind}`.trim();
}

function clearRoots(): void {
  for (const root of roots) {
    scene.remove(root);
    root.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (mesh.geometry) mesh.geometry.dispose();
    });
  }
  roots = [];
}

function frame(): void {
  const box = new THREE.Box3();
  let has = false;
  for (const root of roots) {
    if (root.visible) {
      box.expandByObject(root);
      has = true;
    }
  }
  if (!has) return;
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const dir = new THREE.Vector3(0.6, 0.45, 1).normalize();
  const dist = (sphere.radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.45;
  orbit.target.copy(sphere.center);
  camera.position.copy(sphere.center).addScaledVector(dir, Math.max(dist, 0.4));
  camera.near = Math.max(dist / 100, 0.001);
  camera.far = dist * 100;
  camera.updateProjectionMatrix();
  orbit.update();
}

async function loadSamples(): Promise<void> {
  try {
    const res = await fetch('/api/samples');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    samples = data.samples ?? [];
    const prev = dom.sampleSelect.value;
    dom.sampleSelect.innerHTML = '<option value="">— select sample —</option>';
    for (const s of samples) {
      const opt = document.createElement('option');
      opt.value = s.id;
      opt.textContent = s.id;
      dom.sampleSelect.appendChild(opt);
    }
    if (samples.some((s) => s.id === prev)) dom.sampleSelect.value = prev;
    dom.sampleHint.textContent = samples.length ? `${samples.length} sample(s).` : 'No samples in dataset/.';
  } catch (err) {
    dom.sampleHint.textContent = `Failed to list samples: ${(err as Error).message}`;
  }
}

/** Anchor first (file named `<id>.glb`), then the rest — purely from filenames. */
function orderGlbs(id: string, glbs: string[]): string[] {
  const anchor = `${id}.glb`;
  return [...glbs].sort((a, b) => {
    if (a === anchor) return -1;
    if (b === anchor) return 1;
    return a.localeCompare(b);
  });
}

async function loadSample(id: string): Promise<void> {
  const sample = samples.find((s) => s.id === id);
  if (!sample || sample.glbs.length === 0) {
    setStatus('No .glb files in this sample.', 'err');
    return;
  }

  clearRoots();
  setStatus(`Loading ${id}…`, 'busy');
  dom.selInfo.textContent = 'Loading…';

  const ordered = orderGlbs(id, sample.glbs);
  try {
    const gltfs = await Promise.all(
      ordered.map((name) => loader.loadAsync(`/dataset/${encodeURIComponent(id)}/${encodeURIComponent(name)}`)),
    );
    // Add each GLB exactly as authored — no position/rotation/scale applied here.
    roots = gltfs.map((g, i) => {
      g.scene.name = ordered[i];
      return g.scene;
    });
    scene.add(...roots);

    dom.toggleA.classList.toggle('active', true);
    dom.toggleB.classList.toggle('active', true);
    dom.toggleB.disabled = roots.length < 2;

    frame();

    const labels = ordered.map((n, i) => `${i === 0 ? 'A' : 'B'}: ${n}`);
    dom.fileInfo.textContent = labels.join('\n');
    dom.selInfo.innerHTML =
      `<span class="tag tag-a">A</span> ${ordered[0]}` +
      (ordered[1] ? ` &nbsp; <span class="tag tag-b">B</span> ${ordered[1]}` : '');
    setStatus('Loaded raw — relative pose comes only from the GLB files.', 'ok');
  } catch (err) {
    setStatus(`Load failed: ${(err as Error).message}`, 'err');
  }
}

dom.reloadSamples.addEventListener('click', loadSamples);

dom.sampleSelect.addEventListener('change', () => {
  const id = dom.sampleSelect.value;
  if (id) {
    void loadSample(id);
  } else {
    clearRoots();
    dom.fileInfo.textContent = '';
    dom.selInfo.textContent = 'Pick a sample';
    setStatus('');
  }
});

dom.toggleA.addEventListener('click', () => {
  const root = roots[0];
  if (!root) return;
  root.visible = !root.visible;
  dom.toggleA.classList.toggle('active', root.visible);
});

dom.toggleB.addEventListener('click', () => {
  const root = roots[1];
  if (!root) return;
  root.visible = !root.visible;
  dom.toggleB.classList.toggle('active', root.visible);
});

dom.frameView.addEventListener('click', frame);

window.addEventListener('keydown', (e) => {
  const t = e.target as HTMLElement;
  if (t?.tagName === 'INPUT' || t?.tagName === 'TEXTAREA' || t?.tagName === 'SELECT') return;
  if (e.key.toLowerCase() === 'f') frame();
});

const params = new URLSearchParams(location.search);
void loadSamples().then(() => {
  const id = params.get('sample');
  if (id && samples.some((s) => s.id === id)) {
    dom.sampleSelect.value = id;
    void loadSample(id);
  }
});
