// Part segmentation, both halves of it: what PartField has already cut, and a mesh you drop in
// to have cut now.
//
// The thing worth understanding about the model is that it does not return a segmentation. It
// clusters a learned feature field under the mesh's own face adjacency, and clustering that way
// produces a hierarchy — every number of parts from one to twenty, nested. Which of them is
// right is a property of the shape rather than of the model: eight flatters a chair and shreds
// a mug into meaningless wedges. So the parts slider is the main control on this page, and the
// server answers a level it has not fetched yet in a few seconds off the tree the job already
// built, without going near a GPU.

import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { addStudioLighting } from './viewer';

type Cut = { parts: number; part_faces: number[] };

type Result = {
  id: string;
  source: string;
  job: string;
  faces: number;
  vertices: number;
  cuts: Cut[];
  levels: number[];
  params: { max_clusters: number; option: number; with_knn: boolean; n_point_per_face: number };
  sampling: { n_point_per_face: number; n_sample_each: number };
  on_disk: number[];
};

type Job = { status: 'running' | 'done' | 'failed'; stage: string; error: string | null };

type Part = { node: THREE.Object3D; offset: THREE.Vector3; faces: number; colour: string; visible: boolean };

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node as T;
}

const dom = {
  viewport: el<HTMLDivElement>('viewport'),
  sampleSelect: el<HTMLSelectElement>('sampleSelect'),
  rescan: el<HTMLButtonElement>('rescan'),
  drop: el<HTMLSpanElement>('drop'),
  file: el<HTMLInputElement>('file'),
  dropLabel: el<HTMLSpanElement>('dropLabel'),
  levelSelect: el<HTMLSelectElement>('levelSelect'),
  generateLevel: el<HTMLInputElement>('generateLevel'),
  generateBtn: el<HTMLButtonElement>('generateBtn'),
  levelHint: el<HTMLParagraphElement>('levelHint'),
  explode: el<HTMLInputElement>('explode'),
  showAll: el<HTMLButtonElement>('showAll'),
  frameView: el<HTMLButtonElement>('frameView'),
  parts: el<HTMLDivElement>('parts'),
  numbers: el<HTMLDivElement>('numbers'),
  selInfo: el<HTMLDivElement>('selInfo'),
  status: el<HTMLParagraphElement>('status'),
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

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 5000);
camera.position.set(1.2, 0.9, 1.6);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;

const loader = new GLTFLoader();

let result: Result | null = null;
let root: THREE.Group | null = null;
let parts: Part[] = [];
let level = 8;
let busy = false;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const hex = ([r, g, b]: [number, number, number]) =>
  `#${[r, g, b].map((v) => v.toString(16).padStart(2, '0')).join('')}`;

function setStatus(text: string, kind: '' | 'ok' | 'err' | 'busy' = ''): void {
  dom.status.textContent = text;
  dom.status.className = `status ${kind}`;
}

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error((body as { error?: string }).error ?? `${url} → ${res.status}`);
  return body as T;
}

// ── scene ──────────────────────────────────────────────────────────────────────────────────

function clear(): void {
  if (root) scene.remove(root);
  root = null;
  parts = [];
}

/**
 * The colour the service gave a part, read back off the geometry carrying it.
 *
 * The record names it too, but the file is the thing being drawn and a folder written by an
 * older run may not carry the field at all — so the swatch beside a part and the part itself
 * are answered by one source rather than two that can disagree.
 */
function colourOf(node: THREE.Object3D): string {
  const meshes: THREE.Mesh[] = [];
  node.traverse((child) => {
    if ((child as THREE.Mesh).isMesh) meshes.push(child as THREE.Mesh);
  });

  for (const mesh of meshes) {
    const colour = mesh.geometry.attributes.color;
    if (!colour) continue;
    // A part is one flat colour, so any vertex of it answers for the whole.
    const channels = [colour.getX(0), colour.getY(0), colour.getZ(0)];
    return hex(channels.map((channel) => Math.round(channel * 255)) as [number, number, number]);
  }
  throw new Error(`${node.name} carries no colour`);
}

/**
 * Each part is its own node in the GLB, named for its index. They are re-materialled here rather
 * than drawn off their baked vertex colours so that hiding one, or dimming the rest to solo it,
 * is a property change instead of a reload.
 */
function build(loaded: THREE.Group, cut: Cut): void {
  clear();
  root = loaded;
  scene.add(root);

  const bounds = new THREE.Box3().setFromObject(root);
  const centre = bounds.getCenter(new THREE.Vector3());

  parts = cut.part_faces.map((faces, index) => {
    const name = `part_${String(index).padStart(2, '0')}`;
    const node = root!.getObjectByName(name);
    if (!node) throw new Error(`parts.glb has no ${name}`);

    const colour = colourOf(node);
    const material = new THREE.MeshStandardMaterial({ color: new THREE.Color(colour), roughness: 0.65, metalness: 0.05 });
    node.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (mesh.isMesh) mesh.material = material;
    });

    // Away from the middle of the shape, so exploding pulls the parts apart rather than along
    // any one axis. A part wrapped around the centre barely moves, which is the honest answer.
    const offset = new THREE.Box3().setFromObject(node).getCenter(new THREE.Vector3()).sub(centre);
    return { node, offset, faces, colour, visible: true };
  });

  applyView();
  frame();
}

function applyView(): void {
  const spread = Number(dom.explode.value) / 100;
  for (const part of parts) {
    part.node.visible = part.visible;
    part.node.position.copy(part.offset).multiplyScalar(spread * 1.5);
  }
  for (const [index, button] of [...dom.parts.children].entries()) {
    button.classList.toggle('active', parts[index]?.visible ?? false);
  }

  const shown = parts.filter((part) => part.visible).length;
  const faces = parts.reduce((total, part) => total + (part.visible ? part.faces : 0), 0);
  dom.selInfo.innerHTML = result
    ? `<b>${result.id}</b> · ${shown}/${parts.length} parts · ${faces.toLocaleString()} faces`
    : 'Pick a mesh';
}

function frame(): void {
  const bounds = new THREE.Box3();
  for (const part of parts) if (part.visible) bounds.expandByObject(part.node);
  if (bounds.isEmpty()) return;

  const sphere = bounds.getBoundingSphere(new THREE.Sphere());
  const direction = new THREE.Vector3(0.6, 0.45, 1).normalize();
  // Against the narrower of the two half-angles, so a tall thin panel — which is what this
  // page is in a split window, and what exploding a shape overflows first — still fits it all.
  const vertical = THREE.MathUtils.degToRad(camera.fov / 2);
  const horizontal = Math.atan(Math.tan(vertical) * camera.aspect);
  const distance = (sphere.radius / Math.sin(Math.min(vertical, horizontal))) * 1.45;
  orbit.target.copy(sphere.center);
  camera.position.copy(sphere.center).addScaledVector(direction, Math.max(distance, 0.4));
  camera.near = Math.max(distance / 100, 0.001);
  camera.far = distance * 100;
  camera.updateProjectionMatrix();
  orbit.update();
}

// ── panels ─────────────────────────────────────────────────────────────────────────────────

function renderParts(): void {
  dom.parts.textContent = '';
  for (const [index, part] of parts.entries()) {
    const button = document.createElement('button');
    button.className = 'model';
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = part.colour;
    const label = document.createElement('span');
    label.textContent = `part ${index} · ${part.faces.toLocaleString()} faces`;
    button.append(swatch, label);
    button.title = `Click to hide, press ${index + 1} to solo`;
    button.addEventListener('click', () => {
      part.visible = !part.visible;
      applyView();
    });
    dom.parts.appendChild(button);
  }
}

/** Only the cuts actually sitting in the sample's folder; anything else has to be generated. */
function renderLevels(): void {
  if (!result) return;
  dom.levelSelect.textContent = '';
  for (const parts of result.on_disk) {
    const option = document.createElement('option');
    option.value = String(parts);
    option.textContent = `${parts} parts`;
    dom.levelSelect.appendChild(option);
  }
  dom.levelSelect.disabled = !result.on_disk.length;
  if (result.on_disk.includes(level)) dom.levelSelect.value = String(level);
}

function renderNumbers(): void {
  if (!result) {
    dom.numbers.textContent = '';
    return;
  }
  const rows: [string, string][] = [
    ['faces', result.faces.toLocaleString()],
    ['vertices', result.vertices.toLocaleString()],
    ['fetched', result.on_disk.join(', ')],
    ['samples/face', String(result.sampling?.n_point_per_face ?? '—')],
    ['adjacency', `option ${result.params?.option}${result.params?.with_knn ? ' +knn' : ''}`],
    ['job', result.job],
  ];
  dom.numbers.innerHTML = rows
    .map(([key, value]) => `<div class="bbox-row"><span>${key}</span><span>${value}</span></div>`)
    .join('');
}

// ── loading ────────────────────────────────────────────────────────────────────────────────

async function loadListings(select?: string): Promise<void> {
  const { samples } = await json<{ samples: { id: string; faces: number; cuts: number[] }[] }>('/api/segments');
  const wanted = select ?? dom.sampleSelect.value;
  dom.sampleSelect.textContent = '';
  for (const sample of samples) {
    const option = document.createElement('option');
    option.value = sample.id;
    option.textContent = `${sample.id} — ${sample.cuts.length} cut${sample.cuts.length === 1 ? '' : 's'}`;
    dom.sampleSelect.appendChild(option);
  }
  if (!samples.length) {
    setStatus('Nothing segmented yet — drop a .glb to start.', '');
    return;
  }
  dom.sampleSelect.value = samples.some((sample) => sample.id === wanted) ? wanted : samples[0].id;
  await selectSample(dom.sampleSelect.value);
}

async function selectSample(id: string): Promise<void> {
  const { result: loaded } = await json<{ result: Result }>(`/api/segments/${encodeURIComponent(id)}`);
  result = loaded;

  const maximum = result.levels?.at(-1) ?? result.params.max_clusters;
  dom.generateLevel.min = '1';
  dom.generateLevel.max = String(maximum);

  // Whichever cut is already here, so opening a sample never spends a call to show something.
  level = result.on_disk.includes(level) ? level : (result.on_disk.at(-1) ?? level);
  dom.generateLevel.value = String(level);
  renderLevels();

  history.replaceState(null, '', `?sample=${encodeURIComponent(id)}`);
  renderNumbers();

  // A mesh with nothing fetched yet is left empty rather than cut on sight: taking a level is
  // the Generate button's job, not something picking a sample does behind the user.
  if (!result.on_disk.length) {
    clear();
    dom.parts.textContent = '';
    applyView();
    setStatus(`${result.id} has no cuts yet — pick a number of parts and press Generate.`, '');
    return;
  }
  await loadCut(level);
}

/** A level already on disk loads straight off it; anything else is cut from the job first. */
async function loadCut(parts: number): Promise<void> {
  if (!result) return;
  level = parts;
  dom.generateLevel.value = String(parts);
  renderLevels();

  if (!result.on_disk.includes(parts)) {
    setStatus(`Cutting ${result.id} to ${parts} parts…`, 'busy');
    const res = await fetch(`/api/segments/${encodeURIComponent(result.id)}/cut`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ parts }),
    });
    // A 409 means this mesh is already being worked on — by another tab, or by the load this
    // one replaced. That is still the answer being waited for, so join it rather than fail.
    if (!res.ok && res.status !== 409) {
      throw new Error(((await res.json()) as { error?: string }).error ?? `cut → ${res.status}`);
    }
    result = await awaitJob(result.id);
    renderNumbers();
    renderLevels();
  }

  setStatus(`Loading ${parts} parts…`, 'busy');
  const url = `/segment-results/${encodeURIComponent(result.id)}/parts_${String(parts).padStart(2, '0')}.glb`;
  const gltf = await loader.loadAsync(url);
  const cut = result.cuts.find((entry) => entry.parts === parts);
  if (!cut) throw new Error(`${result.id} has no record of a ${parts}-part cut`);

  build(gltf.scene, cut);
  renderParts();
  applyView();
  setStatus(`${result.id} · ${parts} parts`, 'ok');
}

/** Follows a server-side job to completion, reporting the stage the service is on. */
async function awaitJob(id: string): Promise<Result> {
  for (;;) {
    await sleep(1500);
    const { result: record, job } = await json<{ result: Result | null; job: Job | null }>(
      `/api/segments/${encodeURIComponent(id)}`,
    );
    if (job?.status === 'failed') throw new Error(job.error ?? 'the job failed');
    if (job?.status === 'running') {
      setStatus(`${id} · ${job.stage}…`, 'busy');
      continue;
    }
    if (!record) throw new Error(`${id} finished without leaving a result`);
    return record;
  }
}

async function segmentFile(file: File): Promise<void> {
  const query = new URLSearchParams({ name: file.name, parts: String(level), max_clusters: '20' });
  const { id } = await json<{ id: string }>(`/api/segments?${query}`, {
    method: 'POST',
    headers: { 'Content-Type': 'model/gltf-binary' },
    body: await file.arrayBuffer(),
  });
  setStatus(`Segmenting ${file.name}…`, 'busy');
  await awaitJob(id);
  await loadListings(id);
}

/** Every entry point is one at a time and reports its own failure, so nothing here throws out. */
async function guard(work: () => Promise<void>): Promise<void> {
  if (busy) return;
  busy = true;
  try {
    await work();
  } catch (err) {
    setStatus((err as Error).message, 'err');
  } finally {
    busy = false;
  }
}

// ── wiring ─────────────────────────────────────────────────────────────────────────────────

dom.rescan.addEventListener('click', () => void guard(() => loadListings()));
dom.sampleSelect.addEventListener('change', () => void guard(() => selectSample(dom.sampleSelect.value)));

dom.levelSelect.addEventListener('change', () => void guard(() => loadCut(Number(dom.levelSelect.value))));

function generate(): void {
  const parts = Number(dom.generateLevel.value);
  const maximum = Number(dom.generateLevel.max);
  if (!Number.isInteger(parts) || parts < 1 || parts > maximum) {
    setStatus(`Pick a number of parts between 1 and ${maximum}.`, 'err');
    return;
  }
  void guard(() => loadCut(parts));
}

dom.generateBtn.addEventListener('click', generate);
dom.generateLevel.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') generate();
});

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

// A drop that misses the zone still lands, rather than the browser navigating to the file.
window.addEventListener('dragover', (event) => event.preventDefault());
window.addEventListener('drop', (event) => {
  event.preventDefault();
  const [file] = event.dataTransfer?.files ?? [];
  if (file) void guard(() => segmentFile(file));
});

window.addEventListener('keydown', (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
  if (event.key === 'f') frame();
  else if (event.key === 'x') {
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

void guard(() => loadListings(new URLSearchParams(location.search).get('sample') ?? undefined));
