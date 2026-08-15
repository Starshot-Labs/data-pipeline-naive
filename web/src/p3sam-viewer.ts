import './style.css';
import './navigation';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { addStudioLighting } from './viewer';

type ResultSummary = {
  id: string;
  source: string;
  job: string;
  faces: number;
  vertices: number;
  input_faces: number;
  input_vertices: number;
  num_parts: number;
  part_faces: number[];
  part_colors: [number, number, number][];
  params: { post_process: boolean; threshold: number; seed: number };
  cleaned_mesh: boolean;
  total_seconds: number;
  created_at: number;
};

type Listing = Pick<ResultSummary, 'id' | 'source' | 'faces' | 'num_parts' | 'created_at'>;
type Job = { status: 'running' | 'done' | 'failed'; stage: string; error: string | null };
type Part = { node: THREE.Object3D; offset: THREE.Vector3; faces: number; colour: string; visible: boolean };

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node as T;
}

const dom = {
  viewport: el<HTMLDivElement>('viewport'),
  resultSelect: el<HTMLSelectElement>('resultSelect'),
  rescan: el<HTMLButtonElement>('rescan'),
  drop: el<HTMLSpanElement>('drop'),
  file: el<HTMLInputElement>('file'),
  dropLabel: el<HTMLSpanElement>('dropLabel'),
  postProcess: el<HTMLInputElement>('postProcess'),
  threshold: el<HTMLInputElement>('threshold'),
  seed: el<HTMLInputElement>('seed'),
  explode: el<HTMLInputElement>('explode'),
  showAll: el<HTMLButtonElement>('showAll'),
  frameView: el<HTMLButtonElement>('frameView'),
  download: el<HTMLAnchorElement>('download'),
  parts: el<HTMLDivElement>('parts'),
  numbers: el<HTMLDivElement>('numbers'),
  selInfo: el<HTMLDivElement>('selInfo'),
  status: el<HTMLParagraphElement>('status'),
  progress: el<HTMLDivElement>('progress'),
  progressStage: el<HTMLElement>('progressStage'),
  progressTime: el<HTMLElement>('progressTime'),
};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x090e15);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.outputColorSpace = THREE.SRGBColorSpace;
dom.viewport.appendChild(renderer.domElement);

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
addStudioLighting(scene);

const grid = new THREE.GridHelper(4, 24, 0x273248, 0x182130);
(grid.material as THREE.Material).transparent = true;
(grid.material as THREE.Material).opacity = 0.45;
scene.add(grid);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 5000);
camera.position.set(1.2, 0.9, 1.6);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
const loader = new GLTFLoader();

let result: ResultSummary | null = null;
let root: THREE.Group | null = null;
let parts: Part[] = [];
let busy = false;
let startedAt = 0;
let progressTimer: number | null = null;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const hex = ([r, g, b]: [number, number, number]) =>
  `#${[r, g, b].map((value) => value.toString(16).padStart(2, '0')).join('')}`;

function setStatus(text: string, kind: '' | 'ok' | 'err' | 'busy' = ''): void {
  dom.status.textContent = text;
  dom.status.className = `status ${kind}`;
}

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error((body as { error?: string }).error ?? `${url} → ${response.status}`);
  return body as T;
}

function showProgress(stage: string): void {
  dom.progress.hidden = false;
  dom.progressStage.textContent = stage;
  if (!startedAt) startedAt = Date.now();
  if (progressTimer === null) {
    progressTimer = window.setInterval(() => {
      const seconds = Math.floor((Date.now() - startedAt) / 1000);
      dom.progressTime.textContent = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
    }, 250);
  }
}

function hideProgress(): void {
  dom.progress.hidden = true;
  startedAt = 0;
  if (progressTimer !== null) window.clearInterval(progressTimer);
  progressTimer = null;
}

function clear(): void {
  if (root) scene.remove(root);
  root = null;
  parts = [];
}

function disposeMaterial(material: THREE.Material | THREE.Material[]): void {
  for (const item of Array.isArray(material) ? material : [material]) item.dispose();
}

function build(loaded: THREE.Group, summary: ResultSummary): void {
  clear();
  root = loaded;
  scene.add(root);
  const bounds = new THREE.Box3().setFromObject(root);
  const centre = bounds.getCenter(new THREE.Vector3());

  parts = summary.part_faces.map((faces, index) => {
    const name = `part_${String(index).padStart(2, '0')}`;
    const node = root!.getObjectByName(name);
    if (!node) throw new Error(`parts.glb has no ${name}`);
    const colour = hex(summary.part_colors[index]);
    const material = new THREE.MeshStandardMaterial({ color: colour, roughness: 0.62, metalness: 0.03 });
    node.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (!mesh.isMesh) return;
      disposeMaterial(mesh.material);
      mesh.material = material;
    });
    const offset = new THREE.Box3().setFromObject(node).getCenter(new THREE.Vector3()).sub(centre);
    return { node, offset, faces, colour, visible: true };
  });

  const size = bounds.getSize(new THREE.Vector3());
  grid.scale.setScalar(Math.max(size.x, size.z, size.y, 0.5));
  grid.position.y = bounds.min.y;
  renderParts();
  applyView();
  frame();
}

function applyView(): void {
  const spread = Number(dom.explode.value) / 100;
  for (const part of parts) {
    part.node.visible = part.visible;
    part.node.position.copy(part.offset).multiplyScalar(spread * 1.6);
  }
  for (const [index, button] of [...dom.parts.children].entries()) {
    button.classList.toggle('active', parts[index]?.visible ?? false);
  }

  const shown = parts.filter((part) => part.visible).length;
  const faces = parts.reduce((total, part) => total + (part.visible ? part.faces : 0), 0);
  dom.selInfo.innerHTML = result
    ? `<b>${result.source}</b> · ${shown}/${parts.length} parts · ${faces.toLocaleString()} faces`
    : 'No result loaded';
}

function frame(): void {
  const bounds = new THREE.Box3();
  for (const part of parts) if (part.visible) bounds.expandByObject(part.node);
  if (bounds.isEmpty()) return;
  const sphere = bounds.getBoundingSphere(new THREE.Sphere());
  const direction = new THREE.Vector3(0.65, 0.5, 1).normalize();
  const vertical = THREE.MathUtils.degToRad(camera.fov / 2);
  const horizontal = Math.atan(Math.tan(vertical) * camera.aspect);
  const distance = (sphere.radius / Math.sin(Math.min(vertical, horizontal))) * 1.35;
  orbit.target.copy(sphere.center);
  camera.position.copy(sphere.center).addScaledVector(direction, Math.max(distance, 0.4));
  camera.near = Math.max(distance / 100, 0.001);
  camera.far = Math.max(distance * 100, 10);
  camera.updateProjectionMatrix();
  orbit.update();
}

function renderParts(): void {
  dom.parts.textContent = '';
  for (const [index, part] of parts.entries()) {
    const button = document.createElement('button');
    button.className = 'model active';
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = part.colour;
    const label = document.createElement('span');
    label.textContent = `part ${index + 1} · ${part.faces.toLocaleString()} faces`;
    button.append(swatch, label);
    button.addEventListener('click', () => {
      part.visible = !part.visible;
      applyView();
    });
    dom.parts.appendChild(button);
  }
}

function renderNumbers(): void {
  if (!result) {
    dom.numbers.textContent = '';
    return;
  }
  const cleaned = result.input_faces !== result.faces
    ? `${result.input_faces.toLocaleString()} → ${result.faces.toLocaleString()}`
    : result.faces.toLocaleString();
  const rows: [string, string][] = [
    ['parts', result.num_parts.toLocaleString()],
    ['faces', cleaned],
    ['vertices', result.vertices.toLocaleString()],
    ['post-process', result.params.post_process ? String(result.params.threshold) : 'off'],
    ['seed', String(result.params.seed)],
    ['remote time', `${result.total_seconds.toFixed(1)}s`],
    ['job', result.job],
  ];
  dom.numbers.innerHTML = rows
    .map(([key, value]) => `<div class="bbox-row"><span>${key}</span><span>${value}</span></div>`)
    .join('');
}

async function loadListings(select?: string): Promise<void> {
  const { samples } = await json<{ samples: Listing[] }>('/api/p3sam');
  const wanted = select ?? dom.resultSelect.value;
  dom.resultSelect.textContent = '';
  for (const sample of samples) {
    const option = document.createElement('option');
    option.value = sample.id;
    option.textContent = `${sample.source} — ${sample.num_parts} parts`;
    dom.resultSelect.appendChild(option);
  }
  dom.resultSelect.disabled = !samples.length;
  if (!samples.length) {
    clear();
    renderParts();
    renderNumbers();
    setStatus('No P3-SAM results yet — drop a mesh to start.');
    return;
  }
  dom.resultSelect.value = samples.some((sample) => sample.id === wanted) ? wanted : samples[0].id;
  await selectResult(dom.resultSelect.value);
}

async function selectResult(id: string): Promise<void> {
  const { result: loaded } = await json<{ result: ResultSummary }>(`/api/p3sam/${encodeURIComponent(id)}`);
  result = loaded;
  history.replaceState(null, '', `?result=${encodeURIComponent(id)}`);
  dom.postProcess.checked = result.params.post_process;
  dom.threshold.value = String(result.params.threshold);
  dom.seed.value = String(result.params.seed);
  dom.download.href = `/p3sam-results/${encodeURIComponent(id)}/parts.glb`;
  dom.download.download = `${id}.glb`;
  setStatus(`Loading ${result.source}…`, 'busy');
  const gltf = await loader.loadAsync(dom.download.href);
  build(gltf.scene, result);
  renderNumbers();
  setStatus(`${result.source} · ${result.num_parts} parts`, 'ok');
}

async function awaitJob(id: string): Promise<ResultSummary> {
  for (;;) {
    await sleep(1500);
    const { result: record, job } = await json<{ result: ResultSummary | null; job: Job | null }>(
      `/api/p3sam/${encodeURIComponent(id)}`,
    );
    if (job?.status === 'failed') throw new Error(job.error ?? 'P3-SAM failed');
    if (job?.status === 'running') {
      showProgress(job.stage || 'working');
      setStatus(`${id} · ${job.stage}…`, 'busy');
      continue;
    }
    if (!record) throw new Error(`${id} finished without leaving a result`);
    return record;
  }
}

async function segmentFile(file: File): Promise<void> {
  const threshold = Number(dom.threshold.value);
  const seed = Number(dom.seed.value);
  if (!(threshold > 0 && threshold <= 1)) throw new Error('Merge threshold must be greater than 0 and at most 1.');
  if (!Number.isInteger(seed)) throw new Error('Random seed must be an integer.');

  dom.dropLabel.textContent = file.name;
  startedAt = Date.now();
  showProgress('Uploading mesh');
  const query = new URLSearchParams({
    name: file.name,
    post_process: String(dom.postProcess.checked),
    threshold: String(threshold),
    seed: String(seed),
  });
  const { id } = await json<{ id: string }>(`/api/p3sam?${query}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body: await file.arrayBuffer(),
  });
  showProgress('Waiting for Modal');
  await awaitJob(id);
  await loadListings(id);
}

async function guard(work: () => Promise<void>): Promise<void> {
  if (busy) return;
  busy = true;
  dom.rescan.disabled = true;
  try {
    await work();
  } catch (err) {
    setStatus((err as Error).message, 'err');
  } finally {
    hideProgress();
    dom.rescan.disabled = false;
    busy = false;
  }
}

dom.rescan.addEventListener('click', () => void guard(() => loadListings()));
dom.resultSelect.addEventListener('change', () => void guard(() => selectResult(dom.resultSelect.value)));
dom.explode.addEventListener('input', applyView);
dom.showAll.addEventListener('click', () => {
  for (const part of parts) part.visible = true;
  applyView();
});
dom.frameView.addEventListener('click', frame);

dom.file.addEventListener('change', () => {
  const [file] = dom.file.files ?? [];
  dom.file.value = '';
  if (file) void guard(() => segmentFile(file));
});
dom.drop.addEventListener('click', () => dom.file.click());
dom.drop.addEventListener('dragover', (event) => {
  event.preventDefault();
  dom.drop.classList.add('over');
});
dom.drop.addEventListener('dragleave', () => dom.drop.classList.remove('over'));
dom.drop.addEventListener('drop', (event) => {
  event.preventDefault();
  event.stopPropagation();
  dom.drop.classList.remove('over');
  const [file] = event.dataTransfer?.files ?? [];
  if (file) void guard(() => segmentFile(file));
});
window.addEventListener('dragover', (event) => event.preventDefault());
window.addEventListener('drop', (event) => {
  event.preventDefault();
  const [file] = event.dataTransfer?.files ?? [];
  if (file) void guard(() => segmentFile(file));
});

window.addEventListener('keydown', (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
  if (event.key.toLowerCase() === 'f') frame();
  else if (event.key.toLowerCase() === 'x') {
    dom.explode.value = Number(dom.explode.value) > 0 ? '0' : '45';
    applyView();
  } else if (event.key === '0') {
    for (const part of parts) part.visible = true;
    applyView();
  } else if (/^[1-9]$/.test(event.key)) {
    const only = Number(event.key) - 1;
    if (only >= parts.length) return;
    const soloed = parts.every((part, index) => part.visible === (index === only));
    for (const [index, part] of parts.entries()) part.visible = soloed || index === only;
    applyView();
  }
});

function resize(): void {
  const { clientWidth: width, clientHeight: height } = dom.viewport;
  if (!width || !height) return;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}
new ResizeObserver(resize).observe(dom.viewport);
resize();

(function animate(): void {
  requestAnimationFrame(animate);
  orbit.update();
  renderer.render(scene, camera);
})();

void guard(() => loadListings(new URLSearchParams(location.search).get('result') ?? undefined));
