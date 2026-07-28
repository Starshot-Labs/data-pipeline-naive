import './style.css';
import * as THREE from 'three';
import { Viewer } from './viewer';
import { worldBox } from './bbox';
import type { Slot, TransformMode } from './types';

const SLOTS: Slot[] = ['A', 'B'];

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing element #${id}`);
  return node as T;
}

const dom = {
  viewport: el<HTMLDivElement>('viewport'),
  selectA: el<HTMLButtonElement>('selectA'),
  selectB: el<HTMLButtonElement>('selectB'),
  modeTranslate: el<HTMLButtonElement>('modeTranslate'),
  modeRotate: el<HTMLButtonElement>('modeRotate'),
  modeScale: el<HTMLButtonElement>('modeScale'),
  spaceToggle: el<HTMLButtonElement>('spaceToggle'),
  snapToggle: el<HTMLButtonElement>('snapToggle'),
  toggleA: el<HTMLButtonElement>('toggleA'),
  toggleB: el<HTMLButtonElement>('toggleB'),
  resetA: el<HTMLButtonElement>('resetA'),
  resetB: el<HTMLButtonElement>('resetB'),
  frameView: el<HTMLButtonElement>('frameView'),
  infoAB: el<HTMLDivElement>('infoAB'),
  status: el<HTMLParagraphElement>('status'),
  selInfo: el<HTMLDivElement>('selInfo'),
};

const uploads: Record<Slot, { drop: HTMLElement; input: HTMLInputElement; label: HTMLElement; info: HTMLElement }> = {
  A: { drop: el('dropA'), input: el<HTMLInputElement>('fileA'), label: el('nameA'), info: el('infoA') },
  B: { drop: el('dropB'), input: el<HTMLInputElement>('fileB'), label: el('nameB'), info: el('infoB') },
};

const viewer = new Viewer(dom.viewport, {
  onSelect: updateSelInfo,
  onChange: refresh,
  onLoad: refresh,
  onMode: updateModeButtons,
});

if (import.meta.env.DEV) Object.assign(window as unknown as Record<string, unknown>, { viewer, THREE });

let snapOn = false;
let spaceLocal = false;

// --- uploads ---------------------------------------------------------------

async function loadInto(slot: Slot, file: File): Promise<void> {
  const url = URL.createObjectURL(file);
  setStatus(`Loading ${file.name}…`, 'busy');
  try {
    await viewer.loadModel(slot, url, file.name, { keepPose: true });
    uploads[slot].label.textContent = file.name;
    setStatus(`Loaded ${file.name} as ${slot}.`, 'ok');
  } catch (err) {
    setStatus(`Failed to load ${file.name}: ${(err as Error).message}`, 'err');
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** `first` takes file one; a second file goes to the other slot, so a two-file drop fills both. */
async function accept(first: Slot, files: File[]): Promise<void> {
  const models = files.filter((file) => /\.(glb|gltf)$/i.test(file.name));
  if (!models.length) {
    setStatus('Only .glb and .gltf files can be loaded.', 'err');
    return;
  }
  const order: Slot[] = first === 'A' ? ['A', 'B'] : ['B', 'A'];
  for (const [i, file] of models.slice(0, order.length).entries()) await loadInto(order[i], file);
}

// --- readouts --------------------------------------------------------------

const fmt = (n: number) => n.toFixed(3);

const axes = (v: THREE.Vector3) =>
  `<span class="dim-x">${fmt(v.x)}</span> <span class="dim-y">${fmt(v.y)}</span> <span class="dim-z">${fmt(v.z)}</span>`;

const row = (label: string, value: string) => `<div class="bbox-row"><span>${label}</span><span>${value}</span></div>`;

function renderSlot(slot: Slot): void {
  const object = viewer.getObject(slot);
  const head = `<div class="bbox-head">${slot} · ${viewer.getName(slot) ?? 'empty'}</div>`;
  if (!object) {
    uploads[slot].info.innerHTML = head + row('—', '');
    return;
  }
  const euler = new THREE.Euler().setFromQuaternion(object.quaternion);
  const turned = new THREE.Vector3(euler.x, euler.y, euler.z).multiplyScalar(THREE.MathUtils.RAD2DEG);
  const box = worldBox(object);
  uploads[slot].info.innerHTML =
    head +
    row('size', axes(box.getSize(new THREE.Vector3()))) +
    row('center', axes(box.getCenter(new THREE.Vector3()))) +
    row('rotated°', axes(turned)) +
    row('scaled', axes(object.scale));
}

function refresh(): void {
  for (const slot of SLOTS) renderSlot(slot);

  const objA = viewer.getObject('A');
  const objB = viewer.getObject('B');
  const union = objA && objB ? worldBox(objA).union(worldBox(objB)) : null;
  dom.infoAB.innerHTML =
    '<div class="bbox-head">A + B</div>' +
    (union ? row('size', axes(union.getSize(new THREE.Vector3()))) : row('—', ''));

  dom.toggleA.textContent = objA?.visible === false ? 'Show A' : 'Hide A';
  dom.toggleB.textContent = objB?.visible === false ? 'Show B' : 'Hide B';
}

function updateSelInfo(slot: Slot | null): void {
  dom.selInfo.innerHTML = slot
    ? `Selected <span class="tag ${slot === 'A' ? 'tag-a' : 'tag-b'}">${slot}</span> ${viewer.getName(slot) ?? ''}`
    : 'Nothing selected';
  dom.selectA.classList.toggle('active', slot === 'A');
  dom.selectB.classList.toggle('active', slot === 'B');
}

function updateModeButtons(mode: TransformMode): void {
  dom.modeTranslate.classList.toggle('active', mode === 'translate');
  dom.modeRotate.classList.toggle('active', mode === 'rotate');
  dom.modeScale.classList.toggle('active', mode === 'scale');
}

function setStatus(text: string, kind: '' | 'ok' | 'err' | 'busy' = ''): void {
  dom.status.textContent = text;
  dom.status.className = `status ${kind}`.trim();
}

// --- wiring ----------------------------------------------------------------

for (const slot of SLOTS) {
  const { drop, input } = uploads[slot];

  input.addEventListener('change', () => {
    const files = [...(input.files ?? [])];
    input.value = '';
    void accept(slot, files);
  });

  drop.addEventListener('dragover', (event) => {
    event.preventDefault();
    drop.classList.add('over');
  });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (event) => {
    event.preventDefault();
    event.stopPropagation();
    drop.classList.remove('over');
    void accept(slot, [...(event.dataTransfer?.files ?? [])]);
  });
}

// A drop that misses both zones still lands, rather than the browser opening the file.
window.addEventListener('dragover', (event) => event.preventDefault());
window.addEventListener('drop', (event) => {
  event.preventDefault();
  void accept('A', [...(event.dataTransfer?.files ?? [])]);
});

dom.selectA.addEventListener('click', () => viewer.select('A'));
dom.selectB.addEventListener('click', () => viewer.select('B'));

dom.modeTranslate.addEventListener('click', () => viewer.setMode('translate'));
dom.modeRotate.addEventListener('click', () => viewer.setMode('rotate'));
dom.modeScale.addEventListener('click', () => viewer.setMode('scale'));

dom.spaceToggle.addEventListener('click', () => {
  spaceLocal = !spaceLocal;
  viewer.setSpace(spaceLocal ? 'local' : 'world');
  dom.spaceToggle.textContent = `Space: ${spaceLocal ? 'Local' : 'World'}`;
});

dom.snapToggle.addEventListener('click', () => {
  snapOn = !snapOn;
  viewer.setSnap(snapOn);
  dom.snapToggle.textContent = `Snap: ${snapOn ? 'On' : 'Off'}`;
  dom.snapToggle.classList.toggle('active', snapOn);
});

dom.toggleA.addEventListener('click', () => { viewer.toggleVisible('A'); refresh(); });
dom.toggleB.addEventListener('click', () => { viewer.toggleVisible('B'); refresh(); });
dom.resetA.addEventListener('click', () => viewer.resetTransform('A'));
dom.resetB.addEventListener('click', () => viewer.resetTransform('B'));
dom.frameView.addEventListener('click', () => viewer.frame());

refresh();
updateSelInfo(null);
