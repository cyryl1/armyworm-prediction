# Pest Detection API

A production-ready FastAPI backend for pest detection using YOLOv8 object detection model. This API is designed to detect armyworm larvae and other pests in agricultural images with high accuracy and efficiency.

## Features

- **YOLOv8 Integration**: Uses ultralytics YOLOv8 for state-of-the-art object detection
- **Efficient Model Loading**: Singleton pattern ensures model is loaded once at startup
- **RESTful API**: Clean, documented endpoints with OpenAPI/Swagger support
- **Image Upload**: Multipart form-data support for image uploads
- **Structured Responses**: JSON responses with detection results, confidence scores, and bounding boxes
- **Health Checks**: Built-in health endpoint to monitor API status
- **Class Listing**: Endpoint to retrieve supported pest classes
- **Error Handling**: Comprehensive error handling with appropriate HTTP status codes
- **Logging**: Detailed logging for debugging and monitoring
- **CORS Support**: Cross-Origin Resource Sharing enabled for frontend integration
- **Temporary File Management**: Automatic cleanup of uploaded files after processing

## Project Structure

```
armyworm-prediction/
├── app/
│   ├── __init__.py           # Package initialization
│   ├── main.py              # FastAPI application and endpoints
│   ├── detector.py          # YOLO model loading and inference
│   ├── schemas.py           # Pydantic response models
│   └── utils.py             # Helper functions for file handling
├── model/
│   └── best.pt              # YOLOv8 trained model (add your model here)
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Requirements

- Python 3.8+
- YOLO model file (`model/best.pt`)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd armyworm-prediction
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

Place your trained YOLOv8 model file in the `model/` directory:

```bash
mkdir -p model
# Copy your best.pt file to model/best.pt
cp /path/to/your/best.pt model/best.pt
```

## Running the API

### Development

```bash
python -m uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### Production

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### 1. Health Check

**GET** `/health`

Check the health status of the API and model loading status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Get Supported Classes

**GET** `/classes`

Retrieve list of supported pest classes from the model.

**Response:**
```json
{
  "classes": ["fall-armyworm-larva", "fall-armyworm-adult", "other-pest"],
  "count": 3
}
```

### 3. Detect Pests

**POST** `/detect`

Upload an image and get pest detection results.

**Parameters:**
- `file` (required): Image file (multipart/form-data)
  - Supported formats: JPG, JPEG, PNG, BMP, GIF, WebP
- `confidence` (optional): Confidence threshold (default: 0.5)
  - Range: 0.0 to 1.0

**Request Example:**
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@image.jpg" \
  -F "confidence=0.5"
```

**Response:**
```json
{
  "detections": [
    {
      "class_name": "fall-armyworm-larva",
      "confidence": 0.92,
      "bbox": [150.5, 200.3, 280.7, 350.2]
    },
    {
      "class_name": "fall-armyworm-adult",
      "confidence": 0.87,
      "bbox": [500.1, 150.0, 600.5, 250.8]
    }
  ],
  "processing_time_ms": 245.32
}
```

### 4. API Documentation

**Interactive API Docs:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Usage Examples

### Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Get supported classes
response = requests.get("http://localhost:8000/classes")
print(response.json())

# Detect pests
with open("image.jpg", "rb") as f:
    files = {"file": f}
    params = {"confidence": 0.5}
    response = requests.post("http://localhost:8000/detect", files=files, params=params)
    detections = response.json()
    print(detections)
```

### JavaScript/Node.js

```javascript
// Detect pests
const formData = new FormData();
formData.append("file", imageFile);
formData.append("confidence", 0.5);

const response = await fetch("http://localhost:8000/detect", {
  method: "POST",
  body: formData
});

const detections = await response.json();
console.log(detections);
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Get classes
curl http://localhost:8000/classes

# Detect pests
curl -X POST "http://localhost:8000/detect" \
  -F "file=@image.jpg" \
  -F "confidence=0.5"
```

## Architecture

### Singleton Pattern for Model

The `PestDetector` class uses the singleton pattern to ensure the YOLO model is loaded only once at application startup. This avoids the overhead of reloading the model on every request:

```python
detector = PestDetector()
detector.load_model("model/best.pt")

# Model is reused across all requests
```

### Lifecycle Management

FastAPI's lifespan context manager handles:
- **Startup**: Load model and create temp directory
- **Shutdown**: Clean up temporary files

### Error Handling

- **400 Bad Request**: Invalid file type, missing file, invalid confidence value
- **500 Internal Server Error**: Model loading issues, inference failures
- Detailed error messages in JSON responses

## Performance Considerations

1. **Model Caching**: Model loaded once at startup, reused for all requests
2. **Efficient Inference**: YOLOv8 optimized for speed without sacrificing accuracy
3. **Async File Handling**: Non-blocking file operations
4. **Temporary File Cleanup**: Automatic deletion after processing
5. **CORS Middleware**: Configured for optimal cross-origin requests

## Logging

The API logs important events:

- Model loading status
- Request processing
- Inference results
- Errors and exceptions
- File cleanup operations

Logs can be configured via the `logging` module in `app/main.py`.

## Security Considerations

- **File Type Validation**: Only accepted image formats allowed
- **File Size Limits**: Implement via reverse proxy or middleware if needed
- **Confidence Threshold**: Validated to ensure valid range
- **CORS Configuration**: Currently open; restrict origins in production
- **Model Protection**: Model file excluded from git via `.gitignore`

## Troubleshooting

### Model Not Loading

1. Check `model/best.pt` exists
2. Verify model file is a valid YOLOv8 model
3. Check logs for detailed error messages

```bash
python -c "from ultralytics import YOLO; YOLO('model/best.pt')"
```

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Port Already in Use

Change the port:
```bash
python -m uvicorn app.main:app --port 8001
```

### Out of Memory

Use quantized models or reduce inference batch size. Consider:
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
