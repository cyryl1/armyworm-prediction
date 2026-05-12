# Armyworm Pest Detection System

A production-ready FastAPI backend for detecting fall armyworms and maize diseases using YOLOv8 deep learning model. This system processes uploaded images and returns pest detections with confidence scores, bounding boxes, and actionable management recommendations.

## Features

- **Fast Inference**: YOLOv8 model loaded once at startup for optimal performance
- **Multiple Pest Classes**: Detects fall armyworm (egg, frass, larva, damage) and maize diseases
- **Actionable Recommendations**: Each detection includes domain-specific pest management advice
- **Robust File Handling**: Secure image upload with validation and cleanup
- **Error Handling**: Comprehensive error handling for invalid inputs and processing failures
- **Production-Ready**: Structured logging, efficient resource management, and clean API design

## Tech Stack

- **Framework**: FastAPI
- **Model**: YOLOv8 (Ultralytics)
- **Server**: Uvicorn
- **Image Processing**: OpenCV
- **Validation**: Pydantic

## Setup Instructions

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Model

Ensure `model/best.pt` exists in the project directory (YOLO model file).

### 4. Run Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or for production:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`

## API Documentation

### Interactive Docs

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### `GET /`
Returns a simple message to confirm the API is running.

**Response:**
```json
{
  "message": "Pest Detection API running"
}
```

#### `GET /health`
Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### `GET /classes`
Returns all supported pest classes and their IDs.

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

#### `POST /detect`
Detects pests in an uploaded image.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Parameter**: `file` (image file)

**Response:**
```json
{
  "detections": [
    {
      "class_id": 2,
      "class_name": "fall-armyworm-larva",
      "confidence": 0.91,
      "bbox": [150, 200, 300, 350],
      "recommendation": "Apply neem-based pesticide immediately"
    },
    {
      "class_id": 4,
      "class_name": "healthy-maize",
      "confidence": 0.85,
      "bbox": [400, 100, 600, 400],
      "recommendation": "No action needed"
    }
  ]
}
```

## Example Usage

### Using curl

```bash
# Test if API is running
curl http://localhost:8000/

# Get supported classes
curl http://localhost:8000/classes

# Check health
curl http://localhost:8000/health

# Detect pests in an image
curl -X POST -F "file=@path/to/image.jpg" http://localhost:8000/detect
```

### Using Python requests

```python
import requests

# Upload image and get detections
with open("maize_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/detect", files=files)
    detections = response.json()
    
    for detection in detections["detections"]:
        print(f"{detection['class_name']}: {detection['confidence']:.2%}")
        print(f"Recommendation: {detection['recommendation']}")
```

### Using JavaScript/Fetch

```javascript
const formData = new FormData();
formData.append("file", imageFile);

const response = await fetch("http://localhost:8000/detect", {
  method: "POST",
  body: formData
});

const data = await response.json();
console.log(data.detections);
```

## Project Structure

```
backend/
├── app/
│   ├── __init__.py           # Package initialization
│   ├── main.py               # FastAPI application & endpoints
│   ├── detector.py           # Model loading & inference logic
│   ├── schemas.py            # Pydantic response models
│   └── utils.py              # File handling utilities
├── model/
│   └── best.pt               # YOLOv8 trained model
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Class Labels

| ID | Class Name | Category |
|----|-----------|----------|
| 0 | fall-armyworm-egg | Pest |
| 1 | fall-armyworm-frass | Pest Indicator |
| 2 | fall-armyworm-larva | Pest |
| 3 | fall-armyworm-larval-damage | Damage |
| 4 | healthy-maize | Healthy |
| 5 | maize-streak-disease | Disease |

## Pest Management Recommendations

- **fall-armyworm-egg**: Monitor egg clusters, consider removal or targeted spraying
- **fall-armyworm-frass**: Inspect surrounding leaves for larvae
- **fall-armyworm-larva**: Apply neem-based pesticide immediately
- **fall-armyworm-larval-damage**: Monitor crop and apply targeted control
- **healthy-maize**: No action needed
- **maize-streak-disease**: Use resistant maize varieties, control vector (leafhopper)

## Performance Notes

- **Model Loading**: The YOLOv8 model is loaded once at application startup (not per request) for optimal performance
- **Inference Speed**: Typically <500ms per image on GPU, <2s on CPU
- **Memory**: ~2GB RAM with model loaded
- **Concurrency**: Can handle multiple requests thanks to Uvicorn's async architecture

## Error Handling

The API handles the following error cases:

- **400 Bad Request**: No file uploaded or empty file
- **415 Unsupported Media Type**: Invalid image format
- **500 Internal Server Error**: Model inference failure

All errors include descriptive messages to aid debugging.

## Logging

The application logs:
- API request details
- Detection results
- Errors with full tracebacks
- Performance metrics

Logs are printed to console and can be redirected to files using Uvicorn options.

## Development

### Running Tests

```bash
# Install pytest
pip install pytest pytest-asyncio httpx

# Run tests (when test suite is added)
pytest
```

### Code Style

The code follows PEP 8 conventions and includes:
- Type hints for all functions
- Comprehensive docstrings
- Error handling and validation
- Logging at appropriate levels

## Troubleshooting

### Model not found error
Ensure `model/best.pt` exists in the project directory.

### CUDA out of memory
Reduce model batch size or use CPU: Set `device="cpu"` in detector.py

### Image processing error
Verify the uploaded image is a valid format (JPG, PNG, etc.)

## Future Enhancements

- Batch processing endpoint for multiple images
- Confidence threshold customization
- Image annotation output (with bounding boxes)
- Database integration for detection history
- WebSocket support for real-time processing
- Model versioning and A/B testing

## License

Proprietary - Pest Detection System

## Support

For issues or questions, please contact the development team.
