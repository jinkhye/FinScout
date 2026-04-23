from __future__ import annotations

from functools import lru_cache

from .core.config import Settings, get_settings
from .services.document_ingestion_service import DocumentIngestionService


@lru_cache(maxsize=1)
def get_document_ingestion_service() -> DocumentIngestionService:
    return DocumentIngestionService(get_settings())


def get_app_settings() -> Settings:
    return get_settings()
