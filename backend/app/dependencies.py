from __future__ import annotations

from functools import lru_cache

from .core.config import Settings, get_settings
from .services.agent.agent_service import AgentService
from .services.document_processing.document_ingestion_service import (
    DocumentIngestionService,
)
from .services.query_planning.query_context_service import QueryContextService
from .services.query_planning.query_planner_service import QueryPlannerService
from .services.vector_ingestion.vector_ingestion_service import VectorIngestionService
from .services.vector_ingestion.vector_query_service import VectorQueryService


@lru_cache(maxsize=1)
def get_document_ingestion_service() -> DocumentIngestionService:
    return DocumentIngestionService(get_settings())


@lru_cache(maxsize=1)
def get_vector_ingestion_service() -> VectorIngestionService:
    return VectorIngestionService(get_settings())


@lru_cache(maxsize=1)
def get_vector_query_service() -> VectorQueryService:
    return VectorQueryService(get_settings())


@lru_cache(maxsize=1)
def get_query_planner_service() -> QueryPlannerService:
    return QueryPlannerService(get_settings())


@lru_cache(maxsize=1)
def get_query_context_service() -> QueryContextService:
    return QueryContextService(get_settings())


@lru_cache(maxsize=1)
def get_agent_service() -> AgentService:
    settings = get_settings()
    return AgentService(
        settings=settings,
        planner=get_query_planner_service(),
        context_loader=get_query_context_service(),
        vector_query=get_vector_query_service(),
    )


def get_app_settings() -> Settings:
    return get_settings()
