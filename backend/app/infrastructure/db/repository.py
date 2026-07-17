"""Concrete repository — SQLAlchemy implementation of ScoutingSessionRepositoryInterface."""

from __future__ import annotations

from typing import List

from sqlalchemy.orm import Session

from app.domain.entities import ScoutingSessionEntity
from app.domain.interfaces.repository import ScoutingSessionRepositoryInterface
from app.infrastructure.db.models import ScoutingSessionModel


class SQLAlchemyScoutingSessionRepository(ScoutingSessionRepositoryInterface):
    """Adapter that persists scouting sessions via SQLAlchemy."""

    def __init__(self, db: Session) -> None:
        self._db = db

    # ---- interface implementation ----

    def create(self, session: ScoutingSessionEntity) -> ScoutingSessionEntity:
        db_obj = ScoutingSessionModel(
            total_plants=session.total_plants,
            healthy_plants=session.healthy_plants,
            infested_plants=session.infested_plants,
            latitude=session.latitude,
            longitude=session.longitude,
            primary_recommendation=session.primary_recommendation,
        )
        self._db.add(db_obj)
        self._db.commit()
        self._db.refresh(db_obj)

        return ScoutingSessionEntity(
            id=db_obj.id,
            total_plants=db_obj.total_plants,
            healthy_plants=db_obj.healthy_plants,
            infested_plants=db_obj.infested_plants,
            latitude=db_obj.latitude,
            longitude=db_obj.longitude,
            primary_recommendation=db_obj.primary_recommendation,
            created_at=db_obj.created_at.isoformat() if db_obj.created_at else "",
        )

    def list(self, skip: int = 0, limit: int = 100) -> List[ScoutingSessionEntity]:
        rows = (
            self._db.query(ScoutingSessionModel)
            .order_by(ScoutingSessionModel.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )
        return [
            ScoutingSessionEntity(
                id=r.id,
                total_plants=r.total_plants,
                healthy_plants=r.healthy_plants,
                infested_plants=r.infested_plants,
                latitude=r.latitude,
                longitude=r.longitude,
                primary_recommendation=r.primary_recommendation,
                created_at=r.created_at.isoformat() if r.created_at else "",
            )
            for r in rows
        ]
