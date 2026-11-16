# Maize Guard — Frontend

React + Vite + Tailwind single-page app for the pest detection API.

## Pages

- `/` Diagnostic Hub: upload or capture a photo and view detections
- `/stream` Live View: real-time detection over WebSocket
- `/scout` Scouting Mode: count plants and save sessions

## Development

```bash
pnpm install
pnpm dev        # dev server
pnpm build      # production build
pnpm lint
```

Set `VITE_API_ENDPOINT` to the backend URL (default `http://localhost:8000`; see `.env.docker`). Design references live in `original_html/`.
