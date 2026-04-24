from __future__ import annotations

from fastapi import APIRouter

from .routes.agent import router as agent_router
from .routes.documents import router as documents_router
from .routes.query import router as query_router
from .routes.vector import router as vector_router


api_router = APIRouter()
api_router.include_router(agent_router, prefix="/agent", tags=["agent"])
api_router.include_router(documents_router, prefix="/documents", tags=["documents"])
api_router.include_router(query_router, prefix="/query", tags=["query"])
api_router.include_router(vector_router, prefix="/vector", tags=["vector"])
