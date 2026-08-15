import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import type { Slot, TransformMode } from './types';

interface ViewerCallbacks {
  onSelect?: (slot: Slot | null) => void;
  onChange?: () => void;
  onLoad?: (slot: Slot, name: string) => void;
  onMode?: (mode: TransformMode) => void;
}

interface SlotState {
  object: THREE.Object3D;
  name: string;
  homePosition: THREE.Vector3;
}

interface LoadOptions {
  /** Show the pose baked into the GLB, instead of parking the second model beside the first. */
  keepPose?: boolean;
}

const UP = new THREE.Vector3(0, 1, 0);

export function addStudioLighting(scene: THREE.Scene): void {
  const hemi = new THREE.HemisphereLight(0xffffff, 0x404050, 0.6);
  hemi.name = '__hemi';
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(4, 6, 5);
  key.name = '__key';
  const fill = new THREE.DirectionalLight(0xffffff, 0.6);
  fill.position.set(-5, 2, -3);
  fill.name = '__fill';
  scene.add(hemi, key, fill);
}

export class Viewer {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;
  environment: THREE.Texture;

  private readonly container: HTMLElement;
  private readonly orbit: OrbitControls;
  private readonly transform: TransformControls;
  private readonly transformHelper: THREE.Object3D;
  private readonly grid: THREE.GridHelper;
  private readonly loader = new GLTFLoader();
  private readonly raycaster = new THREE.Raycaster();
  private readonly pointer = new THREE.Vector2();

  private readonly slots: Partial<Record<Slot, SlotState>> = {};
  private selected: Slot | null = null;
  private exportRenderer: THREE.WebGLRenderer | null = null;

  constructor(container: HTMLElement, private readonly cb: ViewerCallbacks = {}) {
    this.container = container;
    const { clientWidth: w, clientHeight: h } = container;

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(this.renderer.domElement);

    this.scene.background = new THREE.Color(0x0d1117);

    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environment = this.environment;

    addStudioLighting(this.scene);

    this.grid = new THREE.GridHelper(20, 20, 0x2a313c, 0x1c2230);
    (this.grid.material as THREE.Material).transparent = true;
    (this.grid.material as THREE.Material).opacity = 0.6;
    this.scene.add(this.grid);

    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 5000);
    this.camera.position.set(3, 2, 4);

    this.orbit = new OrbitControls(this.camera, this.renderer.domElement);
    this.orbit.enableDamping = true;
    this.orbit.dampingFactor = 0.08;

    this.transform = new TransformControls(this.camera, this.renderer.domElement);
    this.transform.addEventListener('dragging-changed', (e) => { this.orbit.enabled = !e.value; });
    this.transform.addEventListener('objectChange', () => this.cb.onChange?.());
    this.transformHelper = this.transform.getHelper();
    this.scene.add(this.transformHelper);

    this.renderer.domElement.addEventListener('pointerdown', this.onPointerDown);
    window.addEventListener('keydown', this.onKeyDown);
    new ResizeObserver(() => this.resize()).observe(container);

    this.animate();
  }

  private animate = (): void => {
    requestAnimationFrame(this.animate);
    this.orbit.update();
    this.renderer.render(this.scene, this.camera);
  };

  private resize(): void {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (w === 0 || h === 0) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  // --- loading -------------------------------------------------------------

  async loadModel(slot: Slot, url: string, name: string, { keepPose = false }: LoadOptions = {}): Promise<void> {
    const gltf = await this.loader.loadAsync(url);
    const root = new THREE.Group();
    root.name = `slot_${slot}`;
    root.userData.slot = slot;
    root.add(gltf.scene);

    this.removeSlot(slot);

    const home = new THREE.Vector3(0, 0, 0);
    if (keepPose) {
      // Move the pivot onto the geometry without moving the geometry, so the gizmo lands on
      // the object instead of wherever the file happens to put its origin.
      const center = new THREE.Box3().setFromObject(root).getCenter(new THREE.Vector3());
      gltf.scene.position.sub(center);
      home.copy(center);
    } else if (slot === 'B') {
      const radius = new THREE.Box3().setFromObject(root).getBoundingSphere(new THREE.Sphere()).radius || 1;
      const other = this.slots.A;
      const otherRadius = other
        ? new THREE.Box3().setFromObject(other.object).getBoundingSphere(new THREE.Sphere()).radius
        : radius;
      home.set((otherRadius + radius) * 1.25, 0, 0);
    }
    root.position.copy(home);

    this.slots[slot] = { object: root, name, homePosition: home.clone() };
    this.scene.add(root);

    this.select(slot);
    this.frame();
    this.cb.onLoad?.(slot, name);
    this.cb.onChange?.();
  }

  private removeSlot(slot: Slot): void {
    const state = this.slots[slot];
    if (!state) return;
    if (this.transform.object === state.object) this.transform.detach();
    this.scene.remove(state.object);
    state.object.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (mesh.geometry) mesh.geometry.dispose();
    });
    delete this.slots[slot];
  }

  getObject(slot: Slot): THREE.Object3D | undefined {
    return this.slots[slot]?.object;
  }

  getName(slot: Slot): string | undefined {
    return this.slots[slot]?.name;
  }

  hasBoth(): boolean {
    return !!this.slots.A && !!this.slots.B;
  }

  // --- selection & gizmo ---------------------------------------------------

  private onPointerDown = (event: PointerEvent): void => {
    if (event.button !== 0 || this.transform.axis) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const roots = (Object.keys(this.slots) as Slot[]).map((s) => this.slots[s]!.object);
    const hits = this.raycaster.intersectObjects(roots, true);
    if (!hits.length) { this.select(null); return; }
    let node: THREE.Object3D | null = hits[0].object;
    while (node && node.userData.slot === undefined) node = node.parent;
    this.select((node?.userData.slot as Slot) ?? null);
  };

  private onKeyDown = (event: KeyboardEvent): void => {
    const target = event.target as HTMLElement;
    if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT')) return;
    switch (event.key.toLowerCase()) {
      case 'w': this.setMode('translate'); break;
      case 'e': this.setMode('rotate'); break;
      case 'r': this.setMode('scale'); break;
      case 'f': this.frame(); break;
      case 'a': this.select('A'); break;
      case 'b': this.select('B'); break;
      case 'escape': this.select(null); break;
    }
  };

  select(slot: Slot | null): void {
    const state = slot ? this.slots[slot] : undefined;
    if (slot && state) {
      this.transform.attach(state.object);
      this.selected = slot;
    } else {
      this.transform.detach();
      this.selected = null;
    }
    this.cb.onSelect?.(this.selected);
  }

  getSelected(): Slot | null {
    return this.selected;
  }

  setMode(mode: TransformMode): void {
    this.transform.setMode(mode);
    this.cb.onMode?.(mode);
  }

  getMode(): TransformMode {
    return this.transform.getMode() as TransformMode;
  }

  setSpace(space: 'world' | 'local'): void {
    this.transform.setSpace(space);
  }

  setSnap(enabled: boolean): void {
    this.transform.setTranslationSnap(enabled ? 0.1 : null);
    this.transform.setRotationSnap(enabled ? THREE.MathUtils.degToRad(15) : null);
    this.transform.setScaleSnap(enabled ? 0.1 : null);
  }

  toggleVisible(slot: Slot): boolean {
    const state = this.slots[slot];
    if (!state) return false;
    state.object.visible = !state.object.visible;
    return state.object.visible;
  }

  resetTransform(slot: Slot): void {
    const state = this.slots[slot];
    if (!state) return;
    state.object.position.copy(state.homePosition);
    state.object.quaternion.identity();
    state.object.scale.set(1, 1, 1);
    this.cb.onChange?.();
  }

  frame(): void {
    const box = new THREE.Box3();
    let has = false;
    for (const s of Object.keys(this.slots) as Slot[]) {
      const state = this.slots[s];
      if (state && state.object.visible) { box.expandByObject(state.object); has = true; }
    }
    if (!has) return;
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const dir = new THREE.Vector3(0.6, 0.4, 1).normalize();
    const dist = (sphere.radius / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2))) * 1.4;
    this.orbit.target.copy(sphere.center);
    this.camera.position.copy(sphere.center).addScaledVector(dir, Math.max(dist, 0.5));
    this.camera.near = Math.max(dist / 100, 0.001);
    this.camera.far = dist * 100;
    this.camera.updateProjectionMatrix();
    this.orbit.update();
  }

  // --- offscreen capture ---------------------------------------------------

  private ensureExportRenderer(): THREE.WebGLRenderer {
    if (this.exportRenderer) return this.exportRenderer;
    const r = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    r.setPixelRatio(1);
    r.toneMapping = this.renderer.toneMapping;
    r.toneMappingExposure = this.renderer.toneMappingExposure;
    r.outputColorSpace = this.renderer.outputColorSpace;
    r.setClearColor(0x000000, 0);
    this.exportRenderer = r;
    return r;
  }

  renderOffscreenSync(scene: THREE.Scene, camera: THREE.PerspectiveCamera, width: number, height: number): void {
    const r = this.ensureExportRenderer();
    r.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    r.render(scene, camera);
  }

  readExportBlob(): Promise<Blob> {
    const canvas = this.ensureExportRenderer().domElement;
    return new Promise((resolve, reject) => {
      canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('toBlob returned null'))), 'image/png');
    });
  }

  /** Capture the live scene from a given camera, temporarily hiding the given slots and all helpers. */
  async captureLive(camera: THREE.PerspectiveCamera, width: number, height: number, hide: Slot[]): Promise<Blob> {
    const hiddenState = hide.map((s) => {
      const obj = this.slots[s]?.object;
      const prev = obj?.visible ?? false;
      if (obj) obj.visible = false;
      return { obj, prev };
    });
    const gridPrev = this.grid.visible;
    const helperPrev = this.transformHelper.visible;
    const bgPrev = this.scene.background;
    this.grid.visible = false;
    this.transformHelper.visible = false;
    this.scene.background = null;

    this.renderOffscreenSync(this.scene, camera, width, height);

    this.scene.background = bgPrev;
    this.grid.visible = gridPrev;
    this.transformHelper.visible = helperPrev;
    for (const { obj, prev } of hiddenState) if (obj) obj.visible = prev;

    return this.readExportBlob();
  }

  /** A perspective camera placed along `dirToCam` from `target`, backed off to frame a sphere of `radius`. */
  framedCamera(target: THREE.Vector3, radius: number, fovDeg: number, dirToCam: THREE.Vector3): THREE.PerspectiveCamera {
    const cam = new THREE.PerspectiveCamera(fovDeg, 1, 0.01, 5000);
    const dist = (Math.max(radius, 1e-4) / Math.sin(THREE.MathUtils.degToRad(fovDeg / 2))) * 1.25;
    cam.position.copy(target).addScaledVector(dirToCam.clone().normalize(), dist);
    cam.up.copy(UP);
    cam.lookAt(target);
    cam.updateMatrixWorld(true);
    return cam;
  }
}
