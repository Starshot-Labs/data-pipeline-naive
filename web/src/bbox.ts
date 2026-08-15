import * as THREE from 'three';
import type { BoxInfo } from './types';

const EPS = 1e-9;

export function worldBox(obj: THREE.Object3D): THREE.Box3 {
  return new THREE.Box3().setFromObject(obj);
}

export function boxInfo(box: THREE.Box3): BoxInfo {
  const size = box.getSize(new THREE.Vector3());
  const max = Math.max(size.x, size.y, size.z, EPS);
  const round = (v: number) => Number(v.toFixed(6));
  return {
    dimensions: { x: round(size.x), y: round(size.y), z: round(size.z) },
    ratios: { x: round(size.x / max), y: round(size.y / max), z: round(size.z / max) },
  };
}

/** Per-axis ratio of one box's dimensions to another's (numerator / denominator). */
export function dimensionRatio(numerator: THREE.Box3, denominator: THREE.Box3): { x: number; y: number; z: number } {
  const a = numerator.getSize(new THREE.Vector3());
  const b = denominator.getSize(new THREE.Vector3());
  const r = (n: number, d: number) => Number((n / Math.max(d, EPS)).toFixed(6));
  return { x: r(a.x, b.x), y: r(a.y, b.y), z: r(a.z, b.z) };
}
