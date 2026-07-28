import './style.css';
import * as THREE from 'three';
import { Viewer } from './viewer';
import { exportSample } from './exporter';
import { listModels } from './api';
import { worldBox, boxInfo } from './bbox';
import type { BoxInfo, Slot, TransformMode } from './types';

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing element #${id}`);
  return node as T;
}

const dom = {
  viewport: el<HTMLDivElement>('viewport'),
  modelA: el<HTMLSelectElement>('modelA'),
  modelB: el<HTMLSelectElement>('modelB'),
  reloadModels: el<HTMLButtonElement>('reloadModels'),
  modelsHint: el<HTMLParagraphElement>('modelsHint'),
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
  placement: el<HTMLTextAreaElement>('placement'),
  bboxA: el<HTMLDivElement>('bboxA'),
  bboxAB: el<HTMLDivElement>('bboxAB'),
  exportBtn: el<HTMLButtonElement>('exportBtn'),
  status: el<HTMLParagraphElement>('status'),
  selInfo: el<HTMLDivElement>('selInfo'),
};

const viewer = new Viewer(dom.viewport, {
  onSelect: updateSelInfo,
  onChange: refreshReadouts,
  onLoad: refreshReadouts,
  onMode: updateModeButtons,
});

if (import.meta.env.DEV) Object.assign(window as unknown as Record<string, unknown>, { viewer, THREE });

let snapOn = false;
let spaceLocal = false;

// --- model list ------------------------------------------------------------

async function loadModelList(): Promise<void> {
  try {
    const models = await listModels();
    for (const select of [dom.modelA, dom.modelB]) {
      const prev = select.value;
      select.innerHTML = '<option value="">— select —</option>';
      for (const name of models) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
      }
      if (models.includes(prev)) select.value = prev;
    }
    dom.modelsHint.textContent = models.length
      ? `${models.length} model(s) available.`
      : 'No .glb files found. Drop GLBs into the models/ folder and rescan.';
  } catch (err) {
    dom.modelsHint.textContent = `Could not load models: ${(err as Error).message}`;
  }
}

async function onSelectModel(slot: Slot, select: HTMLSelectElement): Promise<void> {
  const name = select.value;
  if (!name) return;
  setStatus(`Loading ${name}…`, 'busy');
  try {
    await viewer.loadModel(slot, `/models/${encodeURI(name)}`, name);
    setStatus('');
  } catch (err) {
    setStatus(`Failed to load ${name}: ${(err as Error).message}`, 'err');
  }
}

// --- readouts --------------------------------------------------------------

function renderBox(target: HTMLElement, title: string, info: BoxInfo | null): void {
  if (!info) {
    target.innerHTML = `<div class="bbox-head">${title}</div><div class="bbox-row"><span>—</span></div>`;
    return;
  }
  const f = (n: number) => n.toFixed(3);
  const dims = `<span class="dim-x">${f(info.dimensions.x)}</span> × <span class="dim-y">${f(info.dimensions.y)}</span> × <span class="dim-z">${f(info.dimensions.z)}</span>`;
  const ratio = `<span class="dim-x">${f(info.ratios.x)}</span> : <span class="dim-y">${f(info.ratios.y)}</span> : <span class="dim-z">${f(info.ratios.z)}</span>`;
  target.innerHTML =
    `<div class="bbox-head">${title}</div>` +
    `<div class="bbox-row"><span>size</span><span>${dims}</span></div>` +
    `<div class="bbox-row"><span>ratio</span><span>${ratio}</span></div>`;
}

function refreshReadouts(): void {
  const objA = viewer.getObject('A');
  const objB = viewer.getObject('B');
  renderBox(dom.bboxA, 'A only', objA ? boxInfo(worldBox(objA)) : null);
  if (objA && objB) {
    const boxAB = worldBox(objA).clone().union(worldBox(objB));
    renderBox(dom.bboxAB, 'A + B', boxInfo(boxAB));
  } else {
    renderBox(dom.bboxAB, 'A + B', null);
  }
  dom.exportBtn.disabled = !viewer.hasBoth();
  dom.toggleA.textContent = objA?.visible === false ? 'Show A' : 'Hide A';
  dom.toggleB.textContent = objB?.visible === false ? 'Show B' : 'Hide B';
}

function updateSelInfo(slot: Slot | null): void {
  if (!slot) {
    dom.selInfo.innerHTML = 'Nothing selected';
  } else {
    const name = viewer.getName(slot) ?? '';
    const cls = slot === 'A' ? 'tag-a' : 'tag-b';
    dom.selInfo.innerHTML = `Selected <span class="tag ${cls}">${slot}</span> ${name}`;
  }
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

dom.modelA.addEventListener('change', () => onSelectModel('A', dom.modelA));
dom.modelB.addEventListener('change', () => onSelectModel('B', dom.modelB));
dom.reloadModels.addEventListener('click', loadModelList);

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

dom.toggleA.addEventListener('click', () => { viewer.toggleVisible('A'); refreshReadouts(); });
dom.toggleB.addEventListener('click', () => { viewer.toggleVisible('B'); refreshReadouts(); });
dom.resetA.addEventListener('click', () => viewer.resetTransform('A'));
dom.resetB.addEventListener('click', () => viewer.resetTransform('B'));
dom.frameView.addEventListener('click', () => viewer.frame());

dom.exportBtn.addEventListener('click', async () => {
  if (!viewer.hasBoth()) return;
  dom.exportBtn.disabled = true;
  setStatus('Exporting…', 'busy');
  try {
    const result = await exportSample(viewer, dom.placement.value.trim());
    setStatus(`Saved ${result.id}\n${result.path}`, 'ok');
  } catch (err) {
    setStatus(`Export failed: ${(err as Error).message}`, 'err');
  } finally {
    dom.exportBtn.disabled = !viewer.hasBoth();
  }
});

THREE.ColorManagement.enabled = true;

loadModelList();
refreshReadouts();
updateSelInfo(null);
