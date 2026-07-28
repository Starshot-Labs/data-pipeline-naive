import * as THREE from 'three';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';
import { Viewer, addStudioLighting } from './viewer';
import { worldBox, boxInfo, dimensionRatio } from './bbox';
import { arrayBufferToBase64, blobToBase64, postExport } from './api';
import type { CameraInfo, ExportResult, SampleMetadata, Trs } from './types';

const IMAGE_SIZE = 1024;
const IMAGE_A_FOV = 35;
const PITCH_DOWN_DEG = 10;

function assetBase(name: string): string {
  const file = name.split(/[\\/]/).pop() ?? name;
  return file.replace(/\.(glb|gltf)$/i, '').replace(/[^a-zA-Z0-9._-]+/g, '_') || 'asset';
}

const round = (v: number) => Number(v.toFixed(6));
const vecArr = (v: THREE.Vector3): [number, number, number] => [round(v.x), round(v.y), round(v.z)];
const quatArr = (q: THREE.Quaternion): [number, number, number, number] => [round(q.x), round(q.y), round(q.z), round(q.w)];

function trsFrom(matrix: THREE.Matrix4): Trs {
  const p = new THREE.Vector3();
  const q = new THREE.Quaternion();
  const s = new THREE.Vector3();
  matrix.decompose(p, q, s);
  return { position: vecArr(p), quaternion: quatArr(q), scale: vecArr(s) };
}

function centeredClone(object: THREE.Object3D, center: THREE.Vector3): THREE.Group {
  const group = new THREE.Group();
  group.position.copy(center).multiplyScalar(-1);
  group.add(object.clone(true));
  group.updateMatrixWorld(true);
  return group;
}

function exportGLB(root: THREE.Object3D): Promise<ArrayBuffer> {
  const exporter = new GLTFExporter();
  return new Promise((resolve, reject) => {
    exporter.parse(
      root,
      (result) => resolve(result as ArrayBuffer),
      (error) => reject(error),
      { binary: true, onlyVisible: false },
    );
  });
}

function cameraInfo(cam: THREE.PerspectiveCamera, pitchDownDeg?: number): CameraInfo {
  return {
    position: vecArr(cam.position),
    quaternion: quatArr(cam.quaternion),
    fov: round(cam.fov),
    aspect: round(cam.aspect),
    ...(pitchDownDeg !== undefined ? { pitch_down_deg: pitchDownDeg } : {}),
  };
}

export async function exportSample(viewer: Viewer, placement: string): Promise<ExportResult> {
  const objA = viewer.getObject('A');
  const objB = viewer.getObject('B');
  if (!objA || !objB) throw new Error('Both GLB A and GLB B must be loaded.');

  viewer.scene.updateMatrixWorld(true);

  // Anchor: bring A's world bounding-box center to the origin with a single translation.
  const centerA = worldBox(objA).getCenter(new THREE.Vector3());

  // --- geometry export -----------------------------------------------------
  const exportRootA = centeredClone(objA, centerA);
  const exportRootB = centeredClone(objB, centerA);
  const [glbA, glbB] = await Promise.all([exportGLB(exportRootA), exportGLB(exportRootB)]);

  // --- image A: canonical front view (+Z, slight upward pitch), B absent ---
  const sceneA = new THREE.Scene();
  addStudioLighting(sceneA);
  sceneA.environment = viewer.environment;
  const imgRootA = centeredClone(objA, centerA);
  sceneA.add(imgRootA);
  const sphereA = worldBox(imgRootA).getBoundingSphere(new THREE.Sphere());
  const pitch = THREE.MathUtils.degToRad(PITCH_DOWN_DEG);
  // +Z front, camera raised slightly above the target so it looks down and reveals the top.
  const dirA = new THREE.Vector3(0, Math.sin(pitch), Math.cos(pitch));
  const camA = viewer.framedCamera(sphereA.center, sphereA.radius, IMAGE_A_FOV, dirA);
  viewer.renderOffscreenSync(sceneA, camA, IMAGE_SIZE, IMAGE_SIZE);
  const imageA = await viewer.readExportBlob();

  // --- image B: B centered in a square, using the user's camera angle, A hidden ---
  const sphereB = worldBox(objB).getBoundingSphere(new THREE.Sphere());
  const dirB = new THREE.Vector3(0, 0, 1).applyQuaternion(viewer.camera.quaternion);
  const camB = viewer.framedCamera(sphereB.center, sphereB.radius, viewer.camera.fov, dirB);
  const imageB = await viewer.captureLive(camB, IMAGE_SIZE, IMAGE_SIZE, ['A']);

  // --- bounding boxes ------------------------------------------------------
  const boxA = worldBox(objA);
  const boxB = worldBox(objB);
  const boxAB = boxA.clone().union(boxB);

  // --- transforms ----------------------------------------------------------
  const T = new THREE.Matrix4().makeTranslation(-centerA.x, -centerA.y, -centerA.z);
  const aExport = T.clone().multiply(objA.matrixWorld);
  const bExport = T.clone().multiply(objB.matrixWorld);

  // B expressed in A's oriented, bbox-centered rigid frame.
  const rotA = new THREE.Quaternion().setFromRotationMatrix(objA.matrixWorld);
  const frameA = new THREE.Matrix4().compose(centerA, rotA, new THREE.Vector3(1, 1, 1));
  const bRelToA = frameA.clone().invert().multiply(objB.matrixWorld);

  // --- names: <assetBase>_<uuid> for the sample folder and every exported file ---
  const uuid = crypto.randomUUID();
  const nameA = viewer.getName('A') ?? 'a';
  const nameB = viewer.getName('B') ?? 'b';
  const aBase = assetBase(nameA);
  let bBase = assetBase(nameB);
  if (bBase === aBase) bBase = `${bBase}_b`;
  const dirName = `${aBase}_${uuid}`;
  const fileNames = {
    aGlb: `${aBase}_${uuid}.glb`,
    bGlb: `${bBase}_${uuid}.glb`,
    aPng: `${aBase}_${uuid}.png`,
    bPng: `${bBase}_${uuid}.png`,
  };

  const metadata: SampleMetadata = {
    created_at: new Date().toISOString(),
    uuid,
    source: { a: nameA, b: nameB },
    placement,
    export_frame: { anchor_center_world: vecArr(centerA) },
    transforms: {
      a_in_export_frame: trsFrom(aExport),
      b_in_export_frame: trsFrom(bExport),
      b_relative_to_a_local: trsFrom(bRelToA),
    },
    bounding_box: {
      a_only: boxInfo(boxA),
      b_only: boxInfo(boxB),
      a_with_b: boxInfo(boxAB),
      a_to_b: dimensionRatio(boxA, boxB),
    },
    images: {
      a_front: { file: fileNames.aPng, width: IMAGE_SIZE, height: IMAGE_SIZE, camera: cameraInfo(camA, PITCH_DOWN_DEG) },
      b_user_view: { file: fileNames.bPng, width: IMAGE_SIZE, height: IMAGE_SIZE, camera: cameraInfo(camB) },
    },
    files: { a: fileNames.aGlb, b: fileNames.bGlb },
  };

  const [glbA64, glbB64, imgA64, imgB64] = await Promise.all([
    Promise.resolve(arrayBufferToBase64(glbA)),
    Promise.resolve(arrayBufferToBase64(glbB)),
    blobToBase64(imageA),
    blobToBase64(imageB),
  ]);

  return postExport({
    dirName,
    placement,
    metadata,
    files: {
      [fileNames.aGlb]: glbA64,
      [fileNames.bGlb]: glbB64,
      [fileNames.aPng]: imgA64,
      [fileNames.bPng]: imgB64,
    },
  });
}
