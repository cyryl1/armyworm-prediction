from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.sql import func
from .database import Base

class ScoutingSession(Base):
    __tablename__ = "scouting_sessions"

    id = Column(Integer, primary_key=True, index=True)
    total_plants = Column(Integer, default=0)
    healthy_plants = Column(Integer, default=0)
    infested_plants = Column(Integer, default=0)
    
    # Optional location data from the user
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    # We can store a summary or recommendation here
    primary_recommendation = Column(String, nullable=True)
    
    # Track when the scouting occurred
    created_at = Column(DateTime(timezone=True), server_default=func.now())
