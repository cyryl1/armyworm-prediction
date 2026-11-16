# Maize Guard — Armyworm Pest Detection

Detects fall armyworm and maize disease in photos or live camera video using a YOLOv8 model, and returns management recommendations (cultural → biological → chemical), with region-specific advice when the image carries GPS data.

| Part | Stack | Docs |
|------|-------|------|
| `backend/` | FastAPI, YOLOv8 (Ultralytics), OpenCV, SQLAlchemy | [backend/README.md](backend/README.md) |
| `frontend/` | React, Vite, Tailwind (pnpm) | [frontend/README.md](frontend/README.md) |

## Features

- **Diagnostic Hub** (`/`): upload or capture a photo; get detections, an annotated image and recommendations.
- **Live View** (`/stream`): real-time detection over a WebSocket.
- **Scouting Mode** (`/scout`): count healthy vs. infested plants and save sessions.
- **Recommendations**: live EPPO lookup (cached) with built-in FAO/CABI/CIMMYT guidance as offline fallback.

## Run with Docker

```bash
docker compose up --build
```

- Frontend: http://localhost (nginx, port 80)
- Backend API: http://localhost:8000 (docs at `/docs`)
- Postgres 15 backs the scouting sessions. Without `DATABASE_URL` the backend falls back to SQLite (`sql_app.db`).

## Run locally

```bash
# Backend (needs model/best.pt)
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
pnpm install
pnpm dev
```

The frontend reads the API URL from `VITE_API_ENDPOINT` (default `http://localhost:8000`).

## Tests and CI

```bash
cd backend && pytest -q
```

GitHub Actions (`.github/workflows/ci.yml`) runs the backend tests and frontend lint. The backend deploys to Render via `render.yaml`.

## Classes

| ID | Class |
|----|-------|
| 0 | fall-armyworm-egg |
| 1 | fall-armyworm-frass |
| 2 | fall-armyworm-larva |
| 3 | fall-armyworm-larval-damage |
| 4 | healthy-maize |
| 5 | maize-streak-disease |
