from __future__ import annotations

from fastapi import APIRouter

from .routes.documents import router as documents_router


api_router = APIRouter()
api_router.include_router(documents_router, prefix="/documents", tags=["documents"])
