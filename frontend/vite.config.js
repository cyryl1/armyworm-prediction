import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Allow temporary HTTPS tunnels (cloudflared) for testing on phones
    allowedHosts: ['.trycloudflare.com'],
  }
})
