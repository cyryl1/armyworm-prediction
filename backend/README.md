# Pest Detection API

A production-ready FastAPI backend for pest detection using YOLOv8 with real-time WebSocket streaming, MongoDB persistence, and structured management recommendations.

## Features

- **Dual Detection Modes**:
  - Mode A: Image upload with EXIF GPS extraction and server-side bbox annotation
  - Mode B: Real-time WebSocket stream with configurable frame throttling
- **MongoDB Persistence**: Detection history with rich query support
- **YOLOv8 Inference**: Singleton pattern for efficient model reuse
- **Structured Recommendations**: Site-specific pest management guidance (cultural → biological → chemical)
- **API Key Authentication**: Secure `/detect`, `/history`, and `/detect/stream` endpoints
- **OpenCV Annotation**: Server-side bounding box drawing on detected images
- **EXIF Metadata**: Automatic GPS coordinate extraction from uploaded images
- **Async WebSockets**: Efficient frame processing with configurable throttling
- **Comprehensive Logging**: Structured event tracking for debugging and monitoring

## Project Structure

```
armyworm-prediction/backend/
├── app/
│   ├── __init__.py               # Package initialization
│   ├── main.py                   # FastAPI app lifecycle and router setup
│   ├── routes.py                 # All endpoint definitions (modular)
│   ├── detector.py               # YOLO model loading and inference
│   ├── recommendations.py        # Structured pest management guidance
│   ├── history_store.py          # MongoDB persistence layer
│   ├── auth.py                   # API key authentication
│   ├── schemas.py                # Pydantic request/response models
│   └── utils.py                  # File handling, EXIF, annotation
├── tests/
│   ├── conftest.py               # Pytest fixtures
│   ├── test_utils.py             # GPS, annotation, file tests
│   ├── test_history_store.py     # MongoDB persistence tests
│   ├── test_auth.py              # Authentication tests
│   └── test_recommendations.py   # Recommendation logic tests
├── model/
│   └── best.pt                   # YOLOv8 trained model
├── data/
│   └── detections.db             # SQLite backup (optional; use MongoDB)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Requirements

- Python 3.9+
- YOLO model file (`model/best.pt`)
- MongoDB instance (or local MongoDB for development)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd armyworm-prediction/backend
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your Model

```bash
mkdir -p model
cp /path/to/your/best.pt model/best.pt
```

### 5. Configure Environment Variables

Create a `.env` file in the `backend/` directory:

```bash
# API Authentication
API_KEY=your-secret-api-key-here

# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017
MONGO_DB=pest_detection
MONGO_COLLECTION=detections

# WebSocket Throttling (seconds between frame processing)
STREAM_THROTTLE_SECONDS=1.5

# Model Configuration
MODEL_PATH=model/best.pt
```

For MongoDB Atlas cloud, use:
```bash
MONGO_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
```

## Running the API

### Development

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Public Endpoints (No Auth Required)

#### GET `/` — Status Check
Root endpoint confirming API is running.

```bash
curl http://localhost:8000/
```

#### GET `/health` — Health Status
Check API and model status.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### GET `/classes` — Supported Classes
List all detectable pest and disease classes.

```bash
curl http://localhost:8000/classes
```

**Response:**
```json
{
  "classes": {
    "0": "fall-armyworm-egg",
    "1": "fall-armyworm-frass",
    "2": "fall-armyworm-larva",
    "3": "fall-armyworm-larval-damage",
    "4": "healthy-maize",
    "5": "maize-streak-disease"
  }
}
```

### Protected Endpoints (Requires `x-api-key` Header)

#### POST `/detect` — Image Upload Detection
Upload an image for pest detection with GPS metadata extraction and bbox annotation.

**Headers:**
```
x-api-key: your-secret-api-key-here
```

**Request:**
```bash
curl -X POST http://localhost:8000/detect \
  -H "x-api-key: your-secret-api-key-here" \
  -F "file=@field_sample.jpg"
```

**Response:**
```json
{
  "detections": [
    {
      "class_id": 2,
      "class_name": "fall-armyworm-larva",
      "confidence": 0.92,
      "bbox": [150.5, 200.3, 280.7, 350.2],
      "recommendation": "Chemical | Apply targeted control promptly using label-approved products.",
      "recommendation_details": {
        "severity": "high",
        "management_tier": "chemical",
        "primary_action": "Apply targeted control promptly using label-approved products.",
        "secondary_action": "Rotate modes of action to reduce resistance pressure."
      },
      "gps_latitude": -17.83,
      "gps_longitude": 24.65,
      "detection_timestamp": "2026-05-13T14:30:45.123456+00:00",
      "image_url": "/temp/abc123_field_sample_annotated_20260513143045.jpg",
      "farm_id": null,
      "user_id": null
    }
  ]
}
```

#### GET `/history` — Detection History
Retrieve stored detections from MongoDB with optional filtering.

**Headers:**
```
x-api-key: your-secret-api-key-here
```

**Query Parameters:**
- `limit` (optional): Maximum number of records to return (default: 100)

**Request:**
```bash
curl http://localhost:8000/history?limit=50 \
  -H "x-api-key: your-secret-api-key-here"
```

**Response:**
```json
{
  "records": [
    {
      "id": "507f1f77bcf86cd799439011",
      "class_id": 2,
      "class_name": "fall-armyworm-larva",
      "confidence": 0.92,
      "bbox": [150.5, 200.3, 280.7, 350.2],
      "recommendation": "Chemical | Apply targeted control promptly...",
      "recommendation_details": {...},
      "gps_latitude": -17.83,
      "gps_longitude": 24.65,
      "detection_timestamp": "2026-05-13T14:30:45.123456+00:00",
      "image_url": "/temp/abc123_annotated.jpg",
      "farm_id": null,
      "user_id": null
    }
  ]
}
```

#### WebSocket `/detect/stream` — Real-Time Stream Detection
Connect a WebSocket to stream video frames for real-time pest detection with configurable throttling.

**Authentication:** Provide `api_key` as query parameter or `x-api-key` header.

**Request Example (Python):**
```python
import asyncio
import websockets
import base64
from pathlib import Path

async def stream_frames():
    uri = "ws://localhost:8000/detect/stream?api_key=your-secret-api-key-here"
    async with websockets.connect(uri) as websocket:
        # Send JPEG frame (binary)
        frame_data = Path("frame.jpg").read_bytes()
        await websocket.send(frame_data)
        
        # Receive detection results
        response = await websocket.recv()
        print(f"Detections: {response}")

asyncio.run(stream_frames())
```

**Request Example (JavaScript):**
```javascript
const ws = new WebSocket('ws://localhost:8000/detect/stream?api_key=your-secret-api-key-here');

ws.onopen = () => {
  // Send frame (binary)
  const canvas = document.querySelector('canvas');
  canvas.toBlob(blob => {
    ws.send(blob);
  }, 'image/jpeg');
};

ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log('Detections:', result.detections);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};
```

**Response Format:**
```json
{
  "detections": [
    {
      "class_id": 2,
      "class_name": "fall-armyworm-larva",
      "confidence": 0.92,
      "bbox": [150.5, 200.3, 280.7, 350.2],
      "recommendation": "Chemical | Apply targeted control promptly...",
      "recommendation_details": {...},
      "detection_timestamp": "2026-05-13T14:30:45.123456+00:00"
    }
  ]
}
```

**Frame Throttling:**
- Frames are processed at most every `STREAM_THROTTLE_SECONDS` (default: 1.5s)
- Intermediate frames are buffered; only the latest is used
- This prevents CPU overload from high frame rates (e.g., 30fps mobile camera)

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Suite

```bash
# GPS extraction and image annotation tests
pytest tests/test_utils.py -v

# MongoDB persistence tests
pytest tests/test_history_store.py -v

# API key authentication tests
pytest tests/test_auth.py -v

# Recommendation logic tests
pytest tests/test_recommendations.py -v
```

### Coverage Report

```bash
pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html
```

## Architecture Overview

### Mode A: Image Upload Flow
1. Client uploads JPEG/PNG image with `x-api-key` header
2. Server validates file and extracts EXIF GPS coordinates
3. YOLOv8 inference runs on the image
4. OpenCV draws bounding boxes on detected objects
5. Recommendations matched to each class
6. All results stored in MongoDB
7. Response includes annotated image path and GPS coordinates

### Mode B: Real-Time WebSocket Stream
1. Client connects WebSocket with API key
2. Client streams video frames (JPEG) at camera rate (e.g., 30fps)
3. Server buffers the latest frame
4. Background task processes frames every `STREAM_THROTTLE_SECONDS`
5. Inference results sent back to client as JSON
6. Client renders bounding boxes on live video canvas

### Authentication Flow
- HTTP endpoints require `x-api-key` header
- WebSocket requires `api_key` query parameter or `x-api-key` header
- Missing or invalid key returns 401/403 or closes WebSocket with code 1008
- API key configured via `API_KEY` environment variable

### MongoDB Schema

**Collection: `detections`**

```javascript
{
  "_id": ObjectId,
  "class_id": Number,
  "class_name": String,
  "confidence": Float (0-1),
  "bbox": [x1, y1, x2, y2],
  "recommendation": String,
  "recommendation_details": {
    "severity": String,
    "management_tier": String,
    "primary_action": String,
    "secondary_action": String
  },
  "gps_latitude": Float,
  "gps_longitude": Float,
  "detection_timestamp": ISODate,
  "image_url": String,
  "farm_id": String,
  "user_id": String
}
```

**Indexes:**
- `detection_timestamp` (descending) — for fast history queries
- `class_name` — for filtering by pest type

## Usage Examples

### Python: Upload Image and Get Detections

```python
import requests
import json

API_KEY = "your-secret-api-key-here"
headers = {"x-api-key": API_KEY}

# Upload image
with open("field_sample.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/detect",
        files=files,
        headers=headers
    )

result = response.json()
for detection in result["detections"]:
    print(f"{detection['class_name']}: {detection['confidence']:.1%}")
    print(f"  Recommendation: {detection['recommendation']}")
    if detection.get("gps_latitude"):
        print(f"  Location: ({detection['gps_latitude']}, {detection['gps_longitude']})")
```

### Python: Query Detection History

```python
import requests

API_KEY = "your-secret-api-key-here"
headers = {"x-api-key": API_KEY}

response = requests.get(
    "http://localhost:8000/history?limit=20",
    headers=headers
)

history = response.json()
for record in history["records"]:
    print(f"{record['detection_timestamp']}: {record['class_name']} ({record['confidence']:.1%})")
```

### JavaScript: Real-Time Stream Detection

```javascript
const API_KEY = "your-secret-api-key-here";
const ws = new WebSocket(`ws://localhost:8000/detect/stream?api_key=${API_KEY}`);

// Get video stream from camera
navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
  .then(stream => {
    const video = document.querySelector('video');
    video.srcObject = stream;
    
    const canvas = document.querySelector('canvas');
    const ctx = canvas.getContext('2d');
    
    // Send frames every 500ms (throttle at 1.5s server-side)
    setInterval(() => {
      ctx.drawImage(video, 0, 0);
      canvas.toBlob(blob => ws.send(blob), 'image/jpeg', 0.8);
    }, 500);
  });

// Handle detections
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  data.detections.forEach(det => {
    // Draw on canvas
    const [x1, y1, x2, y2] = det.bbox;
    ctx.strokeStyle = 'lime';
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = 'lime';
    ctx.font = '12px Arial';
    ctx.fillText(`${det.class_name} ${(det.confidence * 100).toFixed(0)}%`, x1, y1 - 5);
  });
};
```

## Performance Tuning

### GPU Acceleration

```bash
# Check CUDA availability
python -c "from ultralytics import YOLO; YOLO('model/best.pt').to('cuda:0')"

# Set device via environment
export CUDA_VISIBLE_DEVICES=0
```

### MongoDB Connection Pooling

Adjust `MONGO_URI` with connection pool parameters:

```bash
MONGO_URI=mongodb://localhost:27017/?maxPoolSize=50&minPoolSize=10
```

### WebSocket Frame Throttling

Tune throttle rate based on CPU capacity:

```bash
# Process 1 frame per 2 seconds (slower, lower CPU)
STREAM_THROTTLE_SECONDS=2.0

# Process 1 frame per second (faster, higher CPU)
STREAM_THROTTLE_SECONDS=1.0
```

## Security Considerations

- **API Key**: Set a strong random key; rotate regularly
- **HTTPS/WSS**: Use reverse proxy (nginx) for TLS in production
- **CORS**: Configure allowed origins in FastAPI middleware
- **File Upload**: Validate file types and sizes
- **MongoDB**: Use authentication; restrict network access
- **Logging**: Sanitize logs to avoid logging sensitive data

## Troubleshooting

### Model Not Loading
```bash
python -c "from ultralytics import YOLO; YOLO('model/best.pt')"
```

### MongoDB Connection Failed
```bash
# Test MongoDB connectivity
python -c "from pymongo import MongoClient; MongoClient('mongodb://localhost:27017').admin.command('ping')"
```

### WebSocket Connection Refused
- Verify API key is correct
- Check `API_KEY` environment variable is set
- Ensure WebSocket endpoint is not blocked by firewall

### Out of Memory
- Reduce WebSocket throttle rate (process fewer frames)
- Use GPU acceleration for faster inference
- Implement batch size limiting

## Contributing

Fork, create feature branch, ensure tests pass, and submit pull request.

```bash
pytest tests/ --cov=app
```

## License

[Specify your license here]
- Using a smaller YOLOv8 variant (nano, small)
- Reducing image resolution
- Limiting concurrent requests

## Dependencies

- **fastapi**: Web framework
- **uvicorn**: ASGI server
- **python-multipart**: File upload support
- **ultralytics**: YOLO framework
- **opencv-python**: Image processing
- **pydantic**: Data validation

See `requirements.txt` for specific versions.

## Future Enhancements

- [ ] Batch image processing
- [ ] Model management endpoints (reload, switch models)
- [ ] Image preprocessing options
- [ ] Result caching
- [ ] Database integration for detection history
- [ ] Authentication and authorization
- [ ] Rate limiting
- [ ] Model versioning
- [ ] GPU/TPU acceleration
- [ ] WebSocket support for real-time streaming

## Contributing

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Commit changes: `git commit -am 'Add new feature'`
3. Push to branch: `git push origin feature/my-feature`
4. Submit a pull request

## License

[Add your license here]

## Support

For issues and questions:
- Check the troubleshooting section
- Review API logs
- Check FastAPI documentation: https://fastapi.tiangolo.com/
- Check ultralytics documentation: https://docs.ultralytics.com/

## Citation

If you use this project, please cite:

```bibtex
@software{pest_detection_api,
  title={Pest Detection API},
  year={2024},
  url={https://github.com/yourusername/armyworm-prediction}
}
```
