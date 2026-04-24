from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from ...core.config import Settings
from ...core.logger import log_json_artifact
from ...schemas.vector import (
    VectorQueryFilters,
    VectorQueryRequest,
    VectorQueryResponse,
    VectorQueryResult,
)
from ..common.gemini import embed_text_with_retries, get_gemini_client
from .vector_index import resolve_collection_name


class VectorQueryService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def query(
        self,
        request: VectorQueryRequest,
        logger: logging.Logger,
    ) -> VectorQueryResponse:
        try:
            logger.info(
                "Vector query request for %s",
                request.collection_name or request.processed_file_path,
            )
            return await asyncio.to_thread(self._query_sync, request, logger)
        except Exception as exc:
            logger.error("Vector query failed: %s", exc)
            return self._error_response(request, exc, logger)

    def _query_sync(
        self,
        request: VectorQueryRequest,
        logger: logging.Logger,
    ) -> VectorQueryResponse:
        collection_name = resolve_collection_name(
            self._settings,
            request.processed_file_path,
            request.collection_name,
        )
        qdrant = self._connect_to_qdrant(logger)
        if not qdrant.collection_exists(collection_name):
            raise ValueError(f"Qdrant collection not found: {collection_name}")

        gemini_client = get_gemini_client()
        if gemini_client is None:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY must be set for embeddings"
            )

        query_vector = embed_text_with_retries(
            gemini_client=gemini_client,
            model=self._settings.gemini_embedding_model,
            text=self._query_embedding_text(request.query),
            max_retries=self._settings.gemini_embedding_max_retries,
            retry_delay_sec=self._settings.gemini_retry_delay_sec,
            logger=logger,
            label="query",
        )
        qdrant_filter = self._build_filter(request.filters)
        logger.info("Searching %s with top_k=%d", collection_name, request.top_k)

        response = qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=request.top_k,
            with_payload=True,
            with_vectors=False,
        )
        results = self._build_results(getattr(response, "points", []) or [])

        summary = {
            "processed_file_path": request.processed_file_path,
            "collection_name": collection_name,
            "query": request.query,
            "top_k": request.top_k,
            "filters": request.filters.model_dump() if request.filters else None,
            "results_count": len(results),
            "status": "success",
            "errors": [],
        }
        log_json_artifact(logger, "query_summary.json", summary)
        logger.info("Vector query returned %d results", len(results))

        return VectorQueryResponse(
            collection_name=collection_name,
            query=request.query,
            top_k=request.top_k,
            results_count=len(results),
            results=results,
            status="success",
            errors=[],
        )

    def _query_embedding_text(self, query: str) -> str:
        return f"task: question answering | query: {query.strip()}"

    def _connect_to_qdrant(self, logger: logging.Logger) -> QdrantClient:
        qdrant = QdrantClient(url=self._settings.qdrant_url)
        try:
            qdrant.get_collections()
        except Exception as exc:
            raise ConnectionError(
                "Could not connect to local Qdrant at "
                f"{self._settings.qdrant_url}. Start it with: "
                "docker run -p 6333:6333 -p 6334:6334 "
                "-v qdrant_storage:/qdrant/storage qdrant/qdrant"
            ) from exc

        logger.info("Connected to Qdrant at %s", self._settings.qdrant_url)
        return qdrant

    def _build_filter(self, filters: VectorQueryFilters | None) -> Filter | None:
        if filters is None:
            return None

        conditions: List[FieldCondition] = []
        if filters.sections:
            conditions.append(
                FieldCondition(
                    key="section",
                    match=MatchAny(any=list(filters.sections)),
                )
            )
        if filters.company_name:
            conditions.append(
                FieldCondition(
                    key="company_name",
                    match=MatchValue(value=filters.company_name),
                )
            )
        if filters.year:
            conditions.append(
                FieldCondition(key="year", match=MatchValue(value=filters.year))
            )
        if filters.has_table is not None:
            conditions.append(
                FieldCondition(
                    key="has_table",
                    match=MatchValue(value=filters.has_table),
                )
            )

        return Filter(must=conditions) if conditions else None

    def _build_results(self, points: List[Any]) -> List[VectorQueryResult]:
        results: List[VectorQueryResult] = []
        for point in points:
            payload: Dict[str, Any] = getattr(point, "payload", None) or {}
            results.append(
                VectorQueryResult(
                    score=float(getattr(point, "score", 0.0) or 0.0),
                    pdf_name=str(payload.get("pdf_name") or ""),
                    company_name=str(payload.get("company_name") or ""),
                    year=str(payload.get("year") or ""),
                    page_number=payload.get("page_number"),
                    section=str(payload.get("section") or ""),
                    has_table=bool(payload.get("has_table", False)),
                    text=str(payload.get("text") or ""),
                    text_for_embedding=str(payload.get("text_for_embedding") or ""),
                    tables=payload.get("tables") or [],
                )
            )

        return results

    def _error_response(
        self,
        request: VectorQueryRequest,
        exc: Exception,
        logger: logging.Logger,
    ) -> VectorQueryResponse:
        error = str(exc)
        log_json_artifact(
            logger,
            "query_summary.json",
            {
                "collection_name": request.collection_name,
                "processed_file_path": request.processed_file_path,
                "query": request.query,
                "top_k": request.top_k,
                "results_count": 0,
                "status": "error",
                "error": error,
                "errors": [error],
            },
        )
        return VectorQueryResponse(
            collection_name=request.collection_name or "",
            query=request.query,
            top_k=request.top_k,
            status="error",
            error=error,
            errors=[error],
        )
