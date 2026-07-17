"""Use case: Manage scouting sessions.

Provides create and list operations for scouting session records,
delegating persistence to the injected repository.
"""

from __future__ import annotations

import logging
from typing import List

from app.domain.entities import ScoutingSessionEntity
from app.domain.interfaces.repository import ScoutingSessionRepositoryInterface

logger = logging.getLogger(__name__)


class CreateScoutingSessionUseCase:
    """Create and persist a new scouting session."""

    def __init__(self, repo: ScoutingSessionRepositoryInterface) -> None:
        self._repo = repo

    def execute(self, session: ScoutingSessionEntity) -> ScoutingSessionEntity:
        logger.info("Creating new scouting session")
        return self._repo.create(session)


class ListScoutingSessionsUseCase:
    """Retrieve a paginated list of scouting sessions."""

    def __init__(self, repo: ScoutingSessionRepositoryInterface) -> None:
        self._repo = repo

    def execute(self, skip: int = 0, limit: int = 100) -> List[ScoutingSessionEntity]:
        logger.info(f"Listing scouting sessions (skip={skip}, limit={limit})")
        return self._repo.list(skip=skip, limit=limit)
