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

## Edge inference spike (`/bench`)

`/bench` times in-browser YOLO inference (onnxruntime-web, WASM and WebGPU). It needs `public/models/best.onnx`, which is gitignored. Generate it from the backend model:

```bash
docker compose exec backend python -c "from ultralytics import YOLO; YOLO('model/best.pt').export(format='onnx', imgsz=640, opset=12)"
docker cp armyworm_backend:/app/model/best.onnx frontend/public/models/best.onnx
```

The ONNX runtime files are copied to `public/ort` automatically before `dev` and `build`. WebGPU needs HTTPS or localhost.
