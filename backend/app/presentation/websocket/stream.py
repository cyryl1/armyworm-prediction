"""WebSocket stream handler for real-time pest detection.

The live video stream stays synchronous in the main FastAPI process
(using thread executor for blocking YOLO inference) as per the
design decision — no background queue for real-time frames.
"""

from __future__ import annotations

import os
import uuid
import time
import asyncio
import base64
import json
import logging
from datetime import datetime, timezone

import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.infrastructure.ml.yolo_detector import get_detector
from app.infrastructure.external.recommendations import (
    format_recommendation,
    get_recommendation_details_async,
)
from app.infrastructure.utils import (
    get_temp_file_path,
    delete_file,
    annotate_image_array_to_base64,
)

logger = logging.getLogger(__name__)

ws_router = APIRouter()

# Same detector instance as the REST routes; the model is loaded in main.py startup.
_detector = get_detector()


@ws_router.websocket("/detect/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    throttle_sec = float(os.getenv("STREAM_THROTTLE_SECONDS", "0.0"))
    last_processed = 0.0
    last_frame: bytes | None = None
    stopped = False

    logger.info("WebSocket client connected for /detect/stream")

    def _run_inference(frame_bytes: bytes):
        """Run YOLO inference synchronously (called from executor)."""
        temp_name = f"{uuid.uuid4().hex}_stream.jpg"
        temp_path = get_temp_file_path(temp_name)
        try:
            with open(temp_path, "wb") as f:
                f.write(frame_bytes)

            detections = _detector.detect(str(temp_path))

            annotated_b64 = None
            frame_image = cv2.imread(str(temp_path))
            if frame_image is not None:
                annotated_b64 = annotate_image_array_to_base64(frame_image, detections)

            for d in detections:
                d["detection_timestamp"] = datetime.now(timezone.utc).isoformat()

            return {"detections": detections, "annotated_frame": annotated_b64}
        finally:
            try:
                if temp_path.exists():
                    delete_file(temp_path)
            except Exception:
                pass

    loop = asyncio.get_event_loop()

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

                frame_to_process = last_frame
                last_frame = None

                try:
                    result = await loop.run_in_executor(None, _run_inference, frame_to_process)

                    for d in result["detections"]:
                        live_rec = await get_recommendation_details_async(d["class_name"])
                        d["recommendation"] = format_recommendation(d["class_name"])
                        d["recommendation_details"] = live_rec
                        d["data_source"] = live_rec.get("data_source", "offline")
                        d["recommendation_last_updated"] = live_rec.get("recommendation_last_updated")

                    await websocket.send_json(result)
                    last_processed = time.time()
                except Exception as e:
                    logger.error(f"Stream processing error: {e}")
                    try:
                        await websocket.send_json({"error": str(e)})
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
                    parsed = json.loads(data)
                    frame_b64 = parsed.get("frame")
                    if frame_b64:
                        last_frame = base64.b64decode(frame_b64)
                except Exception:
                    try:
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
