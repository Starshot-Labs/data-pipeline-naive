import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import './style.css';
import './internscenes.css';
import './navigation';

interface Manifest {
  scenes: string[];
  assets: Record<string, string>;
}

interface Instance {
  id: number;
  category: string;
  model_uid: string;
  bbox: [number, number, number, number, number, number, number, number, number];
}

const MOVE_KEYS = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE']);
const UP = new THREE.Vector3(0, 1, 0);
const BASE_FLY_SPEED = 4;
const LOOK_RADIANS_PER_PIXEL = THREE.MathUtils.degToRad(82) / 900;
const FINAL_Y_UP = new THREE.Matrix4().makeRotationX(-Math.PI / 2);

class InternSceneViewer {
  private readonly scene = new THREE.Scene();
  private readonly camera: THREE.PerspectiveCamera;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly loader = new GLTFLoader();
  private readonly raycaster = new THREE.Raycaster();
  private readonly pointer = new THREE.Vector2();
  private readonly clock = new THREE.Clock();
  private readonly keys = new Set<string>();
  private readonly selectable: THREE.Object3D[] = [];
  private readonly forward = new THREE.Vector3();
  private readonly right = new THREE.Vector3();
  private readonly movement = new THREE.Vector3();
  private root: THREE.Group | null = null;
  private selectionBounds: THREE.Box3Helper | null = null;
  private rightMouseDown = false;
  private leftPointerId: number | null = null;
  private leftDownX = 0;
  private leftDownY = 0;
  private yaw = 0;
  private pitch = 0;
  private speedLevel = 0;
  private flySpeed = BASE_FLY_SPEED;
  private speedTimer = 0;

  constructor(
    private readonly container: HTMLElement,
    private readonly onSelection: (value: string | null) => void,
    private readonly onSpeed: (speed: number, visible: boolean) => void,
  ) {
    this.scene.background = new THREE.Color(0x111417);
    this.camera = new THREE.PerspectiveCamera(82, 1, 0.02, 2000);
    this.camera.rotation.order = 'YXZ';

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.domElement.tabIndex = 0;
    this.container.appendChild(this.renderer.domElement);

    const room = new RoomEnvironment();
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    const environment = pmrem.fromScene(room, 0.04);
    this.scene.environment = environment.texture;
    room.dispose();
    pmrem.dispose();

    const fill = new THREE.HemisphereLight(0xffffff, 0x9aa7a1, 2.2);
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(8, 14, 10);
    const reverse = new THREE.DirectionalLight(0xdcecff, 1.3);
    reverse.position.set(-10, 6, -8);
    this.scene.add(fill, key, reverse);

    this.renderer.domElement.addEventListener('pointerdown', this.onPointerDown);
    window.addEventListener('pointermove', this.onPointerMove);
    window.addEventListener('pointerup', this.onPointerUp);
    window.addEventListener('pointercancel', this.onPointerCancel);
    this.renderer.domElement.addEventListener('wheel', this.onWheel, { passive: false });
    this.renderer.domElement.addEventListener('contextmenu', this.preventContextMenu);
    this.renderer.domElement.addEventListener('keydown', this.onKeyDown);
    this.renderer.domElement.addEventListener('keyup', this.onKeyUp);
    this.renderer.domElement.addEventListener('blur', this.onBlur);
    document.addEventListener('pointerlockchange', this.onPointerLockChange);
    new ResizeObserver(this.resize).observe(container);
    this.resize();
    this.animate();
  }

  clearSelection(): void {
    if (this.selectionBounds) this.scene.remove(this.selectionBounds);
    this.selectionBounds?.geometry.dispose();
    (this.selectionBounds?.material as THREE.Material | undefined)?.dispose();
    this.selectionBounds = null;
    this.onSelection(null);
  }

  async load(
    sceneId: string,
    files: Map<string, File>,
    manifest: Manifest,
    angleMap: Record<string, number>,
    categoryMap: Record<string, string>,
    progress: (done: number, total: number) => void,
  ): Promise<void> {
    this.clearScene();
    const root = new THREE.Group();
    root.name = sceneId;
    this.root = root;
    this.scene.add(root);

    const sceneBase = `Layout_info/${sceneId}`;
    const layout = await readJson<Instance[]>(requiredFile(files, `${sceneBase}/layout.json`));
    const jobs: Array<() => Promise<void>> = [];

    for (const name of ['floor', 'wall', 'ceiling']) {
      const file = files.get(`${sceneBase}/StructureMesh/${name}.glb`);
      if (file) jobs.push(async () => {
        const object = await this.loadGlb(file);
        object.name = name;
        object.userData.label = name;
        this.prepareMeshes(object, object);
        root.add(object);
      });
    }

    layout.forEach((instance) => {
      if (!instance.model_uid) return;
      const assetPath = manifest.assets[instance.model_uid];
      if (!assetPath) throw new Error(`No asset path for ${instance.model_uid}`);
      jobs.push(async () => {
        const holder = new THREE.Group();
        holder.add(await this.loadGlb(requiredFile(files, assetPath)));
        holder.name = `${instance.category}@${instance.model_uid}`;
        holder.userData.label = instance.category;
        holder.userData.uid = instance.model_uid;
        holder.userData.instanceId = instance.id;
        this.placeInstance(holder, instance, angleMap, categoryMap);
        this.prepareMeshes(holder, holder);
        root.add(holder);
      });
    });

    let done = 0;
    const workers = Array.from({ length: Math.min(8, jobs.length) }, async () => {
      while (jobs.length) {
        const job = jobs.shift();
        if (!job) return;
        await job();
        progress(++done, done + jobs.length);
      }
    });
    await Promise.all(workers);
    root.updateMatrixWorld(true);
    this.frame(root);
  }

  /** A scene already composed by the official InternScenes code, loaded as-is. */
  async loadComposed(source: File | string): Promise<void> {
    this.clearScene();
    const root = new THREE.Group();
    this.root = root;
    this.scene.add(root);

    const object = typeof source === 'string'
      ? (await this.loader.loadAsync(source)).scene
      : await this.loadGlb(source);

    // trimesh exports one "world" root; descend single-child wrappers so each of the
    // composer's object nodes becomes its own selectable root.
    let level: THREE.Object3D = object;
    while (level.children.length === 1 && !(level.children[0] instanceof THREE.Mesh)) {
      level = level.children[0];
    }
    if (!level.children.length) throw new Error('composed scene has no objects');
    for (const child of level.children) {
      const parsed = parseComposedName(child.name);
      child.userData.label = parsed.label;
      if (parsed.uid) child.userData.uid = parsed.uid;
      this.prepareMeshes(child, child);
    }
    root.add(object);
    root.updateMatrixWorld(true);
    this.frame(root);
  }

  private async loadGlb(file: File): Promise<THREE.Object3D> {
    const url = URL.createObjectURL(file);
    try {
      return (await this.loader.loadAsync(url)).scene;
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  /** The canonical pre-rotation the official composer gives each source library, turning a
   *  GLB's Y-up mesh into the layout's Z-up frame with the object facing +X. */
  private canonicalRotation(
    uid: string,
    angleMap: Record<string, number>,
    categoryMap: Record<string, string>,
  ): THREE.Matrix4 {
    if (uid.startsWith('objaverse/')) {
      const angle = angleMap[uid.slice('objaverse/'.length)];
      if (angle === undefined) throw new Error(`Missing Objaverse angle for ${uid}`);
      return rotationZ(THREE.MathUtils.degToRad(angle))
        .multiply(rotationZ(Math.PI / 2))
        .multiply(rotationX(Math.PI / 2));
    }
    if (uid.startsWith('partnet_mobility/')) {
      const matrix = rotationZ(Math.PI).multiply(rotationX(Math.PI / 2));
      if (['Pen', 'Remote', 'Phone'].includes(categoryMap[uid])) {
        return rotationY(Math.PI / 2).multiply(rotationZ(Math.PI)).multiply(matrix);
      }
      return matrix;
    }
    return rotationZ(Math.PI / 2).multiply(rotationX(Math.PI / 2));
  }

  /** The official composer's whole transform chain as one matrix. It has to stay a matrix:
   *  scale is applied between the canonical and layout rotations, so the result cannot be
   *  decomposed into position/rotation/scale — writing those properties silently drops the
   *  canonical rotation and centring, which scatters every object. */
  private placeInstance(
    holder: THREE.Group,
    instance: Instance,
    angleMap: Record<string, number>,
    categoryMap: Record<string, string>,
  ): void {
    const [px, py, pz, sx, sy, sz, rz, rx, ry] = instance.bbox;

    // Bounds are measured per vertex, as trimesh does: Objaverse canonical angles are
    // arbitrary, and a box-corner bound overestimates under rotation, shrinking the object.
    const centroid = new THREE.Box3().setFromObject(holder, true).getCenter(new THREE.Vector3());
    const pre = this.canonicalRotation(instance.model_uid, angleMap, categoryMap)
      .multiply(new THREE.Matrix4().makeTranslation(-centroid.x, -centroid.y, -centroid.z));

    holder.matrixAutoUpdate = false;
    holder.matrix.copy(pre);
    holder.updateMatrixWorld(true);
    const size = new THREE.Box3().setFromObject(holder, true).getSize(new THREE.Vector3());

    const scale = new THREE.Vector3(sx / size.x, sy / size.y, sz / size.z);
    if (instance.category === 'clothes') scale.setScalar(Math.min(scale.x, scale.y, scale.z));
    else if (instance.category === 'carpet') scale.z /= 100;

    // trimesh euler_matrix(a, b, c, axes="rzxy") is Rz(a) · Rx(b) · Ry(c).
    const placed = rotationZ(rz)
      .multiply(rotationX(rx))
      .multiply(rotationY(ry))
      .multiply(new THREE.Matrix4().makeScale(scale.x, scale.y, scale.z));
    placed.setPosition(px, py, pz);

    holder.matrix.copy(FINAL_Y_UP.clone().multiply(placed).multiply(pre));
    holder.updateMatrixWorld(true);
  }

  private prepareMeshes(root: THREE.Object3D, selectableRoot: THREE.Object3D): void {
    root.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return;
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      for (const material of materials) {
        material.side = THREE.DoubleSide;
        if (material instanceof THREE.MeshStandardMaterial) material.envMapIntensity = 0.65;
        material.needsUpdate = true;
      }
      child.userData.selectableRoot = selectableRoot;
      this.selectable.push(child);
    });
  }

  private frame(root: THREE.Object3D): void {
    const bounds = new THREE.Box3().setFromObject(root);
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const eyeHeight = Math.min(Math.max(size.y * 0.18, 1.4), 2.0);
    this.camera.position.set(center.x, bounds.min.y + eyeHeight, bounds.max.z + Math.max(size.z * 0.15, 1.5));
    this.camera.lookAt(center.x, bounds.min.y + eyeHeight, center.z);
    const rotation = new THREE.Euler().setFromQuaternion(this.camera.quaternion, 'YXZ');
    this.pitch = rotation.x;
    this.yaw = rotation.y;
    this.camera.rotation.set(this.pitch, this.yaw, 0, 'YXZ');
    const radius = bounds.getBoundingSphere(new THREE.Sphere()).radius;
    this.camera.near = Math.max(radius / 1000, 0.005);
    this.camera.far = Math.max(radius * 50, 100);
    this.camera.updateProjectionMatrix();
  }

  private select(object: THREE.Object3D): void {
    this.clearSelection();
    const box = new THREE.Box3().setFromObject(object);
    this.selectionBounds = new THREE.Box3Helper(box, 0xffb020);
    this.selectionBounds.renderOrder = 1000;
    this.scene.add(this.selectionBounds);
    const label = String(object.userData.label ?? object.name ?? 'Object');
    const uid = object.userData.uid ? ` · ${object.userData.uid}` : '';
    this.onSelection(`${label}${uid}`);
  }

  private clearScene(): void {
    this.clearSelection();
    this.selectable.length = 0;
    if (!this.root) return;
    this.scene.remove(this.root);
    dispose(this.root);
    this.root = null;
  }

  private pick = (x: number, y: number): void => {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(((x - rect.left) / rect.width) * 2 - 1, -((y - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hit = this.raycaster.intersectObjects(this.selectable, false)[0];
    if (!hit) return this.clearSelection();
    this.select(hit.object.userData.selectableRoot as THREE.Object3D);
  };

  private beginLook(): void {
    this.rightMouseDown = true;
    this.renderer.domElement.classList.add('looking');
    if (document.pointerLockElement !== this.renderer.domElement) void this.renderer.domElement.requestPointerLock();
  }

  private endLook(): void {
    this.rightMouseDown = false;
    this.renderer.domElement.classList.remove('looking');
    if (document.pointerLockElement === this.renderer.domElement) document.exitPointerLock();
  }

  private onPointerDown = (event: PointerEvent): void => {
    this.renderer.domElement.focus();
    if (event.button === 2) {
      event.preventDefault();
      this.beginLook();
    } else if (event.button === 0) {
      this.leftPointerId = event.pointerId;
      this.leftDownX = event.clientX;
      this.leftDownY = event.clientY;
    }
  };

  private onPointerMove = (event: PointerEvent): void => {
    if (!this.rightMouseDown) return;
    this.yaw -= event.movementX * LOOK_RADIANS_PER_PIXEL;
    this.pitch = THREE.MathUtils.clamp(this.pitch - event.movementY * LOOK_RADIANS_PER_PIXEL, -Math.PI / 2 + 0.02, Math.PI / 2 - 0.02);
    this.camera.rotation.set(this.pitch, this.yaw, 0, 'YXZ');
  };

  private onPointerUp = (event: PointerEvent): void => {
    if (event.button === 2) return this.endLook();
    if (event.button !== 0 || event.pointerId !== this.leftPointerId) return;
    const distance = Math.hypot(event.clientX - this.leftDownX, event.clientY - this.leftDownY);
    this.leftPointerId = null;
    if (distance <= 5) this.pick(event.clientX, event.clientY);
  };

  private onPointerCancel = (event: PointerEvent): void => {
    if (event.pointerId === this.leftPointerId) this.leftPointerId = null;
    if (event.button === 2) this.endLook();
  };

  private onPointerLockChange = (): void => {
    if (this.rightMouseDown && document.pointerLockElement !== this.renderer.domElement) this.endLook();
  };

  private onWheel = (event: WheelEvent): void => {
    event.preventDefault();
    this.speedLevel = THREE.MathUtils.clamp(this.speedLevel + (event.deltaY < 0 ? 1 : -1), -6, 14);
    this.flySpeed = BASE_FLY_SPEED * 1.25 ** this.speedLevel;
    this.onSpeed(this.flySpeed, true);
    window.clearTimeout(this.speedTimer);
    this.speedTimer = window.setTimeout(() => this.onSpeed(this.flySpeed, false), 1100);
  };

  private onKeyDown = (event: KeyboardEvent): void => {
    if (MOVE_KEYS.has(event.code) || event.code.startsWith('Shift')) {
      event.preventDefault();
      this.keys.add(event.code);
    }
  };

  private onKeyUp = (event: KeyboardEvent): void => { this.keys.delete(event.code); };
  private onBlur = (): void => { this.keys.clear(); if (this.rightMouseDown) this.endLook(); };
  private preventContextMenu = (event: Event): void => event.preventDefault();

  private resize = (): void => {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / Math.max(height, 1);
    this.camera.updateProjectionMatrix();
  };

  private animate = (): void => {
    const delta = Math.min(this.clock.getDelta(), 0.05);
    this.movement.set(0, 0, 0);
    this.camera.getWorldDirection(this.forward);
    this.right.crossVectors(this.forward, UP).normalize();
    if (this.keys.has('KeyW')) this.movement.add(this.forward);
    if (this.keys.has('KeyS')) this.movement.sub(this.forward);
    if (this.keys.has('KeyD')) this.movement.add(this.right);
    if (this.keys.has('KeyA')) this.movement.sub(this.right);
    if (this.keys.has('KeyE')) this.movement.add(UP);
    if (this.keys.has('KeyQ')) this.movement.sub(UP);
    if (this.movement.lengthSq()) {
      const sprint = this.keys.has('ShiftLeft') || this.keys.has('ShiftRight');
      this.camera.position.addScaledVector(this.movement.normalize(), this.flySpeed * (sprint ? 2.5 : 1) * delta);
    }
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame(this.animate);
  };
}

function rotationX(angle: number): THREE.Matrix4 { return new THREE.Matrix4().makeRotationX(angle); }
function rotationY(angle: number): THREE.Matrix4 { return new THREE.Matrix4().makeRotationY(angle); }
function rotationZ(angle: number): THREE.Matrix4 { return new THREE.Matrix4().makeRotationZ(angle); }

/** Node names from the official composer look like `3_toilet@partnet_mobility/102620`;
 *  GLTFLoader strips `/` and `.` from names, so the uid part comes back slightly mangled
 *  but still identifies the asset. */
function parseComposedName(name: string): { label: string; uid?: string } {
  const match = /^\d+_(.+?)@(.+)$/.exec(name);
  if (!match) return { label: name || 'structure' };
  return { label: match[1].replace(/_/g, ' '), uid: match[2] };
}

function requiredFile(files: Map<string, File>, path: string): File {
  const file = files.get(path);
  if (!file) throw new Error(`Folder is missing ${path}`);
  return file;
}

async function readJson<T>(file: File): Promise<T> { return JSON.parse(await file.text()) as T; }

function dispose(root: THREE.Object3D): void {
  root.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    child.geometry.dispose();
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    for (const material of materials) material.dispose();
  });
}

const viewport = document.querySelector<HTMLElement>('#viewport')!;
const status = document.querySelector<HTMLElement>('#status')!;
const selection = document.querySelector<HTMLElement>('#selection')!;
const speed = document.querySelector<HTMLElement>('#speed')!;
const folder = document.querySelector<HTMLInputElement>('#folder')!;
const picker = document.querySelector<HTMLElement>('#scene-picker')!;
const select = document.querySelector<HTMLSelectElement>('#scene')!;
const clear = document.querySelector<HTMLButtonElement>('#clear')!;

const viewer = new InternSceneViewer(
  viewport,
  (value) => {
    selection.hidden = !value;
    selection.textContent = value ?? '';
    clear.hidden = !value;
  },
  (value, visible) => {
    speed.hidden = !visible;
    speed.textContent = `${value.toFixed(2)} m/s`;
  },
);

let files = new Map<string, File>();
let manifest: Manifest | null = null;
let angleMap: Record<string, number> = {};
let categoryMap: Record<string, string> = {};
let loadGeneration = 0;

folder.addEventListener('change', async () => {
  const chosen = Array.from(folder.files ?? []);
  if (!chosen.length) return;
  const root = chosen[0].webkitRelativePath.split('/')[0];
  files = new Map(chosen.map((file) => [file.webkitRelativePath.slice(root.length + 1), file]));
  try {
    manifest = await readJson<Manifest>(requiredFile(files, 'manifest.json'));
    angleMap = await readJson(requiredFile(files, 'asset_library/uid_2_angle.json'));
    categoryMap = await readJson(requiredFile(files, 'asset_library/uid_2_origin_cate.json'));
    select.replaceChildren(...manifest.scenes.map((scene) => new Option(scene, scene)));
    picker.hidden = manifest.scenes.length < 2;
    await loadSelected();
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : String(error);
    status.classList.add('error');
  }
});

select.addEventListener('change', () => { void loadSelected(); });
clear.addEventListener('click', () => viewer.clearSelection());

async function loadSelected(): Promise<void> {
  if (!manifest) return;
  const generation = ++loadGeneration;
  const sceneId = select.value || manifest.scenes[0];
  status.classList.remove('error');
  status.hidden = false;
  status.textContent = `Loading ${sceneId}…`;
  try {
    const composed = files.get(`composed/${sceneId}/glb_scene.glb`);
    if (composed) {
      await viewer.loadComposed(composed);
      if (generation === loadGeneration) status.hidden = true;
    } else {
      await viewer.load(sceneId, files, manifest, angleMap, categoryMap, (done, total) => {
        if (generation === loadGeneration) status.textContent = `Loading ${sceneId}: ${done}/${total}`;
      });
      if (generation === loadGeneration) {
        status.textContent = 'browser-composed — run pipeline/compose_internscenes.py for the official composition';
        window.setTimeout(() => { if (generation === loadGeneration) status.hidden = true; }, 5000);
      }
    }
  } catch (error) {
    if (generation !== loadGeneration) return;
    status.textContent = error instanceof Error ? error.message : String(error);
    status.classList.add('error');
  }
}

// Debug handle: lets devtools (and headless checks) force renders and move the camera.
(window as unknown as Record<string, unknown>).viewer = viewer;

const composedUrl = new URLSearchParams(location.search).get('glb');
if (composedUrl) {
  status.textContent = `Loading ${composedUrl}…`;
  viewer.loadComposed(composedUrl).then(
    () => { status.hidden = true; },
    (error) => {
      status.textContent = error instanceof Error ? error.message : String(error);
      status.classList.add('error');
    },
  );
}

