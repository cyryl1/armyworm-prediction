import { createReadStream, existsSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In dev, Vite refuses JS imports that resolve into /public, but onnxruntime-web
// dynamically imports its runtime (.mjs) from wasmPaths. Serve /ort straight from
// node_modules, ahead of Vite's own middleware. Builds use public/ort (copy-ort).
function serveOrtRuntime() {
  const dir = fileURLToPath(new URL('./node_modules/onnxruntime-web/dist', import.meta.url))
  const allowed = /^ort-wasm-simd-threaded[\w.-]*\.(mjs|wasm)$/
  return {
    name: 'serve-ort-runtime',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/ort', (req, res, next) => {
        const name = decodeURIComponent((req.url || '').split('?')[0]).replace(/^\//, '')
        const file = join(dir, name)
        if (!allowed.test(name) || !existsSync(file)) return next()
        res.setHeader('Content-Type', name.endsWith('.wasm') ? 'application/wasm' : 'text/javascript')
        createReadStream(file).pipe(res)
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), serveOrtRuntime()],
  server: {
    port: 5173,
    // Allow temporary HTTPS tunnels (cloudflared) for testing on phones
    allowedHosts: ['.trycloudflare.com'],
  },
})
