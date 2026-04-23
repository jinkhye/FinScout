from __future__ import annotations

from fastapi import FastAPI

from .api.router import api_router
from .core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.api_version)
    app.state.settings = settings
    app.include_router(api_router, prefix="/api/v1")

    @app.get("/health")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
