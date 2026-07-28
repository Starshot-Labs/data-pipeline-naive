import type { ExportPayload, ExportResult } from './types';

export async function listModels(): Promise<string[]> {
  const res = await fetch('/api/models');
  if (!res.ok) throw new Error(`Failed to list models (${res.status})`);
  const data = await res.json();
  return data.models ?? [];
}

export async function postExport(payload: ExportPayload): Promise<ExportResult> {
  const res = await fetch('/api/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Export failed (${res.status}): ${text}`);
  }
  return res.json();
}

export function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  const chunk = 0x8000;
  let binary = '';
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

export async function blobToBase64(blob: Blob): Promise<string> {
  return arrayBufferToBase64(await blob.arrayBuffer());
}
