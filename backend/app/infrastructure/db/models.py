"""SQLAlchemy ORM models.

Each class maps to a database table.  These models are infrastructure
concerns — the domain layer uses plain dataclasses instead.
"""

from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.sql import func

from app.infrastructure.db.database import Base


class ScoutingSessionModel(Base):
    """Persisted scouting session record."""

    __tablename__ = "scouting_sessions"

    id = Column(Integer, primary_key=True, index=True)
    total_plants = Column(Integer, default=0)
    healthy_plants = Column(Integer, default=0)
    infested_plants = Column(Integer, default=0)

    # Optional location data from the user
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    # Summary / recommendation
    primary_recommendation = Column(String, nullable=True)

    # Track when the scouting occurred
    created_at = Column(DateTime(timezone=True), server_default=func.now())
