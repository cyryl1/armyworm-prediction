// Copies the onnxruntime-web WASM runtime into public/ort so it is served
// same-origin (and can be cached for offline use). Runs before dev/build.
import { cpSync, mkdirSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = join(root, 'node_modules', 'onnxruntime-web', 'dist');
const dest = join(root, 'public', 'ort');
mkdirSync(dest, { recursive: true });

for (const f of readdirSync(src)) {
  if (/^ort-wasm-simd-threaded.*\.(wasm|mjs)$/.test(f)) cpSync(join(src, f), join(dest, f));
}
console.log('onnxruntime-web runtime copied to public/ort');
