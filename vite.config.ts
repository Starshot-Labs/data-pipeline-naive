import { defineConfig } from 'vite';
import { resolve } from 'node:path';

const backend = process.env.BACKEND_URL ?? 'http://localhost:3000';

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      '/api': backend,
      '/models': backend,
      '/dataset': backend,
      // Without these the dev server answers a sample's meshes and images with its own
      // HTML fallback, and honouring GENERATED_DIR is the backend's job either way.
      '/generated': backend,
      '/placement-results': backend,
      '/edit-results': backend,
      '/segment-results': backend,
      '/mesh': backend,
      '/out': backend,
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        place: resolve(__dirname, 'place.html'),
        viewer: resolve(__dirname, 'viewer.html'),
        pipeline: resolve(__dirname, 'pipeline.html'),
        placement: resolve(__dirname, 'placement.html'),
        edit: resolve(__dirname, 'edit.html'),
        segment: resolve(__dirname, 'segment.html'),
      },
    },
  },
});
