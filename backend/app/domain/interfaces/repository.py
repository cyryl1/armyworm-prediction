"""Port: Scouting session repository interface.

Defines the contract for persisting and retrieving scouting sessions.
Concrete implementations may target PostgreSQL, MongoDB, in-memory, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from app.domain.entities import ScoutingSessionEntity


class ScoutingSessionRepositoryInterface(ABC):
    """Contract for scouting session persistence."""

    @abstractmethod
    def create(self, session: ScoutingSessionEntity) -> ScoutingSessionEntity:
        """Persist a new scouting session and return it with a generated ID."""
        ...

    @abstractmethod
    def list(self, skip: int = 0, limit: int = 100) -> List[ScoutingSessionEntity]:
        """Retrieve scouting sessions ordered by newest first."""
        ...
