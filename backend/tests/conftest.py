from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import pytest
from fastapi.testclient import TestClient

# Ensure repo-root imports like `backend.app...` work when pytest is invoked
# directly from the workspace without extra PYTHONPATH configuration.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.dependencies import get_agent_service
from backend.app.main import create_app
from backend.app.schemas.agent import AgentAskResponse


class StubAgentService:
    def __init__(self, handler: Callable[..., AgentAskResponse]) -> None:
        self._handler = handler

    async def ask(self, request, logger):
        return self._handler(request=request, logger=logger)


@pytest.fixture
def app():
    application = create_app()
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def override_agent_service(app):
    def _apply(handler: Callable[..., AgentAskResponse]) -> None:
        app.dependency_overrides[get_agent_service] = lambda: StubAgentService(handler)

    return _apply
