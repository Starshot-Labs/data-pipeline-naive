export type Slot = 'A' | 'B';
export type TransformMode = 'translate' | 'rotate' | 'scale';

export interface Vec3 { x: number; y: number; z: number; }
export interface Quat { x: number; y: number; z: number; w: number; }

export interface Trs {
  position: [number, number, number];
  quaternion: [number, number, number, number];
  scale: [number, number, number];
}

export interface BoxInfo {
  dimensions: { x: number; y: number; z: number };
  ratios: { x: number; y: number; z: number };
}

export interface CameraInfo {
  position: [number, number, number];
  quaternion: [number, number, number, number];
  fov: number;
  aspect: number;
  pitch_down_deg?: number;
}

export interface SampleMetadata {
  created_at: string;
  uuid: string;
  source: { a: string; b: string };
  placement: string;
  export_frame: { anchor_center_world: [number, number, number] };
  transforms: {
    a_in_export_frame: Trs;
    b_in_export_frame: Trs;
    b_relative_to_a_local: Trs;
  };
  bounding_box: {
    a_only: BoxInfo;
    b_only: BoxInfo;
    a_with_b: BoxInfo;
    a_to_b: { x: number; y: number; z: number };
  };
  images: {
    a_front: { file: string; width: number; height: number; camera: CameraInfo };
    b_user_view: { file: string; width: number; height: number; camera: CameraInfo };
  };
  files: { a: string; b: string };
}

export interface ExportPayload {
  dirName: string;
  placement: string;
  metadata: SampleMetadata;
  files: Record<string, string>;
}

export interface ExportResult { id: string; path: string; }
