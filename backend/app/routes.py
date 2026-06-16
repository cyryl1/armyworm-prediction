"""API route definitions for the pest detection service.

This module centralizes HTTP and WebSocket routes so `app.main` remains
focused on application lifecycle and configuration.
"""

from __future__ import annotations

import os

import cv2
import uuid
import time
import asyncio
import base64
import json
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect

from app import detector, utils
from app.schemas import DetectionResponse, Detection, ClassesResponse, HealthResponse

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", tags=["Status"])
async def root():
    logger.info("GET / - API status check")
    return {"message": "Pest Detection API running"}


@router.get("/health", response_model=HealthResponse, tags=["Status"])
async def health():
    logger.info("GET /health - Health check")
    return HealthResponse(status="healthy", model_loaded=detector.is_model_loaded())


@router.get("/classes", response_model=ClassesResponse, tags=["Information"])
async def get_classes():
    logger.info("GET /classes - Retrieving supported classes")
    classes = detector.get_all_classes()
    classes_str = {str(k): v for k, v in classes.items()}
    return ClassesResponse(classes=classes_str)


@router.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_pests(file: UploadFile = File(...)):
    logger.info(f"POST /detect - Processing image: {file.filename}")

    if not file:
        logger.warning("No file uploaded")
        raise HTTPException(status_code=400, detail="No file uploaded")

    if not utils.validate_file_extension(file.filename):
        logger.warning(f"Invalid file extension: {file.filename}")
        raise HTTPException(status_code=415, detail=f"Invalid file type. Allowed types: {', '.join(utils.ALLOWED_EXTENSIONS)}")

    temp_file_path = None
    try:
        file_content = await file.read()
        if not utils.validate_file_size(len(file_content)):
            logger.warning(f"File size too large: {len(file_content)} bytes")
            raise HTTPException(status_code=400, detail="File size exceeds maximum limit (10MB)")

        unique_filename, temp_file_path = utils.save_uploaded_file(file_content, file.filename)
        logger.info(f"Saved temporary file: {temp_file_path}")

        gps_latitude, gps_longitude = utils.extract_gps_from_exif(temp_file_path)

        detections_list = detector.detect(str(temp_file_path))

        # Generate base64-encoded annotated image with bounding boxes
        annotated_image_b64 = None
        if detections_list:
            annotated_image_b64 = utils.annotate_detections_to_base64(
                temp_file_path, detections_list
            )

        detections: List[Detection] = [
            Detection(
                class_id=d["class_id"],
                class_name=d["class_name"],
                confidence=d["confidence"],
                bbox=d["bbox"],
                recommendation=d["recommendation"],
                recommendation_details=d.get("recommendation_details"),
                gps_latitude=gps_latitude,
                gps_longitude=gps_longitude,
                detection_timestamp=datetime.now(timezone.utc).isoformat(),
            )
            for d in detections_list
        ]

        logger.info(f"Returning {len(detections)} detections")
        return DetectionResponse(
            detections=detections,
            annotated_image=annotated_image_b64,
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"File validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Error processing image")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if temp_file_path:
            utils.delete_file(temp_file_path)


@router.websocket("/detect/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    throttle_sec = float(os.getenv("STREAM_THROTTLE_SECONDS", "1.0"))
    last_processed = 0.0
    last_frame: bytes | None = None
    stopped = False

    logger.info("WebSocket client connected for /detect/stream")

    async def processor():
        nonlocal last_frame, last_processed, stopped
        try:
            while not stopped:
                await asyncio.sleep(0.05)
                if last_frame is None:
                    continue
                now = time.time()
                if now - last_processed < throttle_sec:
                    continue

                temp_name = f"{uuid.uuid4().hex}_stream.jpg"
                temp_path = utils.get_temp_file_path(temp_name)
                try:
                    with open(temp_path, "wb") as f:
                        f.write(last_frame)

                    detections = detector.detect(str(temp_path))

                    # Generate annotated frame with bounding boxes
                    annotated_b64 = None
                    frame_image = cv2.imread(str(temp_path))
                    if frame_image is not None:
                        annotated_b64 = utils.annotate_image_array_to_base64(
                            frame_image, detections
                        )

                    for d in detections:
                        d["detection_timestamp"] = datetime.now(timezone.utc).isoformat()
                    await websocket.send_json({
                        "detections": detections,
                        "annotated_frame": annotated_b64,
                    })
                    last_processed = now
                except Exception as e:
                    logger.error(f"Stream processing error: {e}")
                    try:
                        await websocket.send_json({"error": str(e)})
                    except Exception:
                        pass
                finally:
                    try:
                        if temp_path.exists():
                            utils.delete_file(temp_path)
                    except Exception:
                        pass
        except asyncio.CancelledError:
            return

    processor_task = asyncio.create_task(processor())

    try:
        while True:
            data = await websocket.receive_text()
            if data:
                try:
                    # Try to parse as JSON with base64 frame
                    parsed = json.loads(data)
                    frame_b64 = parsed.get("frame")
                    if frame_b64:
                        last_frame = base64.b64decode(frame_b64)
                except Exception:
                    try:
                        # Try as raw base64
                        last_frame = base64.b64decode(data)
                    except Exception:
                        logger.debug("Received unparseable text frame on websocket stream")
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    finally:
        stopped = True
        processor_task.cancel()
        try:
            await processor_task
        except Exception:
            pass
