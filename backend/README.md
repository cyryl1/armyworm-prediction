# Pest Detection API

FastAPI backend that runs a YOLOv8 model on images and live video frames, enriches detections with pest-management recommendations, and stores scouting sessions.

## Architecture

Clean architecture; dependencies point inward.

```
app/
├── main.py                  # App setup, startup (DB tables, shared model load)
├── domain/                  # Entities (dataclasses) and interfaces (ports)
├── usecases/
│   ├── detect_pests.py      # Validate → temp save → EXIF GPS → YOLO → annotate → recommend
│   └── scouting.py          # Create / list scouting sessions
├── infrastructure/
│   ├── ml/yolo_detector.py  # YOLOv8 adapter + get_detector() shared instance
│   ├── db/                  # SQLAlchemy models, repository (SQLite or Postgres)
│   ├── external/            # EPPO API client, cache, recommendations
│   └── utils.py             # File validation, EXIF GPS, annotation
└── presentation/
    ├── rest/                # Routes and Pydantic schemas
    └── websocket/stream.py  # /detect/stream
tests/                       # pytest
model/best.pt                # YOLOv8 weights
```

A single detector instance is loaded at startup and shared by the REST routes and the WebSocket stream. Detection is synchronous (the stream runs inference in a thread executor); there is no task queue.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Requires `model/best.pt`. Docs: http://localhost:8000/docs

### Environment variables (optional, `backend/.env`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | `sqlite:///./sql_app.db` | Database (Postgres in Docker) |
| `STREAM_THROTTLE_SECONDS` | `0.0` | Min seconds between processed stream frames |
| `LIVE_API_ENABLED` | `true` | Enable live EPPO recommendation lookups |
| `LIVE_API_TIMEOUT_SECONDS` | `3.0` | Timeout for live API calls |
| `EPPO_AUTH_TOKEN` | — | EPPO API token |

There is no authentication; CORS is open to all origins.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Status message |
| GET | `/health` | `{status, model_loaded}` |
| GET | `/classes` | Class ID → name map |
| POST | `/detect` | Multipart `file` (jpg/jpeg/png/gif/bmp/webp, max 10 MB). Returns `detections` and a base64 `annotated_image` |
| POST | `/scouting-sessions` | Save a session (`total_plants`, `healthy_plants`, `infested_plants`, `latitude`, `longitude`, `primary_recommendation`) |
| GET | `/scouting-sessions` | List sessions (`skip`, `limit`) |
| WS | `/detect/stream` | Send frames, receive detections |

### `POST /detect` response

```json
{
  "detections": [
    {
      "class_id": 2,
      "class_name": "fall-armyworm-larva",
      "confidence": 0.92,
      "bbox": [0.12, 0.25, 0.31, 0.48],
      "recommendation": "...",
      "recommendation_details": {},
      "gps_latitude": null,
      "gps_longitude": null,
      "detection_timestamp": "2026-05-13T14:30:45+00:00",
      "data_source": "offline",
      "recommendation_last_updated": null
    }
  ],
  "annotated_image": "<base64 JPEG or null>"
}
```

`bbox` is normalised (0–1). `data_source` is `live` when the recommendation came from EPPO, otherwise `offline`.

### `WS /detect/stream`

Send each frame as a base64 JPEG string, either bare or as JSON `{"frame": "<base64>"}`. Only the latest frame is processed (throttled by `STREAM_THROTTLE_SECONDS`). Each reply is `{"detections": [...], "annotated_frame": "<base64 JPEG>"}`, or `{"error": "..."}` on failure.

## Tests

```bash
pytest -q
```
