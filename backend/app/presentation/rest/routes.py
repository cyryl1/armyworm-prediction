"""REST API route definitions for the pest detection service.

This module defines HTTP endpoints. Detection is synchronous:
  POST /detect        → process image, return DetectionResponse
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from sqlalchemy.orm import Session

from app.domain.entities import ScoutingSessionEntity
from app.infrastructure.db.database import get_db
from app.infrastructure.db.repository import SQLAlchemyScoutingSessionRepository
from app.infrastructure.ml.yolo_detector import get_detector
from app.presentation.rest.schemas import (
    ClassesResponse,
    Detection,
    DetectionResponse,
    HealthResponse,
    ScoutingSession,
    ScoutingSessionCreate,
)
from app.usecases.detect_pests import DetectPestsUseCase
from app.usecases.scouting import CreateScoutingSessionUseCase, ListScoutingSessionsUseCase

logger = logging.getLogger(__name__)

router = APIRouter()

# Shared detector (same instance as the WebSocket stream); loaded in main.py startup.
_detector = get_detector()


# ---------------------------------------------------------------------------
# Status endpoints
# ---------------------------------------------------------------------------

@router.get("/", tags=["Status"])
async def root():
    logger.info("GET / - API status check")
    return {"message": "Pest Detection API running"}


@router.get("/health", response_model=HealthResponse, tags=["Status"])
async def health():
    logger.info("GET /health - Health check")
    return HealthResponse(status="healthy", model_loaded=_detector.is_model_loaded())


@router.get("/classes", response_model=ClassesResponse, tags=["Information"])
async def get_classes():
    logger.info("GET /classes - Retrieving supported classes")
    classes = _detector.get_all_classes()
    classes_str = {str(k): v for k, v in classes.items()}
    return ClassesResponse(classes=classes_str)


# ---------------------------------------------------------------------------
# Detection endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/detect",
    response_model=DetectionResponse,
    tags=["Detection"],
)
async def detect_pests(file: UploadFile = File(...)):
    """Accept an image upload, process it synchronously, and return results."""
    logger.info(f"POST /detect - Processing image: {file.filename}")

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        file_content = await file.read()

        use_case = DetectPestsUseCase(detector=_detector)
        result = use_case.execute(file_content, file.filename)

        detections = [
            Detection(
                class_id=d["class_id"],
                class_name=d["class_name"],
                confidence=d["confidence"],
                bbox=d["bbox"],
                recommendation=d["recommendation"],
                recommendation_details=d.get("recommendation_details"),
                gps_latitude=d.get("gps_latitude"),
                gps_longitude=d.get("gps_longitude"),
                detection_timestamp=d.get("detection_timestamp"),
                data_source=d.get("data_source", "offline"),
                recommendation_last_updated=d.get("recommendation_last_updated"),
            )
            for d in result.get("detections", [])
        ]
        
        return DetectionResponse(
            detections=detections,
            annotated_image=result.get("annotated_image"),
        )

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error processing detection: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------------------------------------------------------------------------
# Scouting session endpoints
# ---------------------------------------------------------------------------

@router.post("/scouting-sessions", response_model=ScoutingSession, tags=["Scouting"])
async def create_scouting_session(
    session: ScoutingSessionCreate,
    db: Session = Depends(get_db),
):
    logger.info("POST /scouting-sessions - Saving new scouting session")
    repo = SQLAlchemyScoutingSessionRepository(db)
    use_case = CreateScoutingSessionUseCase(repo)

    entity = ScoutingSessionEntity(
        total_plants=session.total_plants,
        healthy_plants=session.healthy_plants,
        infested_plants=session.infested_plants,
        latitude=session.latitude,
        longitude=session.longitude,
        primary_recommendation=session.primary_recommendation,
    )
    created = use_case.execute(entity)

    return ScoutingSession(
        id=created.id,
        total_plants=created.total_plants,
        healthy_plants=created.healthy_plants,
        infested_plants=created.infested_plants,
        latitude=created.latitude,
        longitude=created.longitude,
        primary_recommendation=created.primary_recommendation,
        created_at=created.created_at or "",
    )


@router.get("/scouting-sessions", response_model=List[ScoutingSession], tags=["Scouting"])
async def get_scouting_sessions(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    logger.info("GET /scouting-sessions - Retrieving scouting sessions")
    repo = SQLAlchemyScoutingSessionRepository(db)
    use_case = ListScoutingSessionsUseCase(repo)

    entities = use_case.execute(skip=skip, limit=limit)
    return [
        ScoutingSession(
            id=e.id,
            total_plants=e.total_plants,
            healthy_plants=e.healthy_plants,
            infested_plants=e.infested_plants,
            latitude=e.latitude,
            longitude=e.longitude,
            primary_recommendation=e.primary_recommendation,
            created_at=e.created_at or "",
        )
        for e in entities
    ]
