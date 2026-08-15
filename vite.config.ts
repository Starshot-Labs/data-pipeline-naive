import { defineConfig } from 'vite';
import { resolve } from 'node:path';

const backend = process.env.BACKEND_URL ?? 'http://127.0.0.1:3000';
const root = resolve(__dirname, 'web');

// Each page is its own entry point: the client is a set of separate tools rather than one app
// with routes, so nothing is shared between them beyond src/.
const PAGES = ['index', 'place', 'viewer', 'pipeline', 'placement', 'edit', 'segment', 'p3sam', 'scene'];

export default defineConfig({
  root,
  server: {
    port: 5173,
    proxy: {
      '/api': backend,
      '/models': backend,
      '/dataset': backend,
      // Without these the dev server answers a sample's meshes and images with its own
      // HTML fallback, and honouring the directory overrides is the backend's job either way.
      '/generated': backend,
      '/placement-results': backend,
      '/edit-results': backend,
      '/segment-results': backend,
      '/p3sam-results': backend,
      '/scenes': backend,
      '/scene-edits': backend,
      '/mesh': backend,
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: Object.fromEntries(PAGES.map((page) => [page, resolve(root, `${page}.html`)])),
    },
  },
});
