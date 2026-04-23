from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from backend.app.api.router import api_router
    from backend.app.core.config import get_settings
else:
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
