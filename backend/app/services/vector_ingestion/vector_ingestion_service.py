from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ...core.config import Settings
from ...core.logger import log_json_artifact
from ...schemas.vector import VectorIngestResponse
from ..common.gemini import embed_text_with_retries, get_gemini_client
from .vector_index import (
    append_vector_run,
    build_vector_index_record,
    load_processed_payload,
    write_vector_index,
)


class VectorIngestionService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def ingest_processed_file(
        self,
        processed_file_path: str,
        logger: logging.Logger,
    ) -> VectorIngestResponse:
        try:
            logger.info("Vector ingestion request for %s", processed_file_path)
            return await asyncio.to_thread(
                self._ingest_processed_file_sync,
                processed_file_path,
                logger,
            )
        except Exception as exc:
            logger.error("Vector ingestion failed: %s", exc)
            return self._error_response(processed_file_path, exc, logger)

    def _ingest_processed_file_sync(
        self,
        processed_file_path: str,
        logger: logging.Logger,
    ) -> VectorIngestResponse:
        path, processed_payload = load_processed_payload(self._settings, processed_file_path)

        pdf_name = str(processed_payload.get("pdf_name") or path.stem)
        collection_name = self._build_collection_name(pdf_name)
        logger.info("Resolved processed JSON: %s", path)
        logger.info("Target Qdrant collection: %s", collection_name)

        pages, skipped_pages = self._pages_to_embed(processed_payload)
        if not pages:
            raise ValueError("No pages with non-empty text_for_embedding found")

        qdrant = self._connect_to_qdrant(logger)
        gemini_client = get_gemini_client()
        if gemini_client is None:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set for embeddings")

        embeddings = self._embed_pages(gemini_client, processed_payload, pages, logger)
        vector_size = len(embeddings[0])
        points = self._build_points(processed_payload, pages, embeddings)

        self._recreate_collection(qdrant, collection_name, vector_size)
        qdrant.upsert(collection_name=collection_name, points=points)

        record = build_vector_index_record(
            settings=self._settings,
            processed_path=path,
            processed_payload=processed_payload,
            collection_name=collection_name,
            points_inserted=len(points),
            skipped_pages=skipped_pages,
        )
        write_vector_index(path, record)
        append_vector_run(path, record)

        summary = {
            "processed_file_path": str(path),
            "collection_name": collection_name,
            "embedding_model": self._settings.gemini_embedding_model,
            "qdrant_url": self._settings.qdrant_url,
            "qdrant_distance": self._settings.qdrant_distance,
            "points_inserted": len(points),
            "skipped_pages": skipped_pages,
            "status": "success",
            "errors": [],
        }
        log_json_artifact(logger, "ingestion_summary.json", summary)
        logger.info("Inserted %d points into %s", len(points), collection_name)
        logger.info("Saved vector index metadata to %s", path.parent / "vector_index.json")

        return VectorIngestResponse(
            collection_name=collection_name,
            points_inserted=len(points),
            status="success",
            errors=[],
        )

    def _pages_to_embed(
        self,
        processed_payload: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], List[int]]:
        pages: List[Dict[str, Any]] = []
        skipped_pages: List[int] = []

        for page in processed_payload.get("pages", []):
            if not isinstance(page, dict):
                continue

            page_number = int(page.get("page_number") or 0)
            text_for_embedding = str(page.get("text_for_embedding") or "").strip()
            if not text_for_embedding:
                skipped_pages.append(page_number)
                continue

            pages.append(page)

        return pages, skipped_pages

    def _embed_pages(
        self,
        gemini_client: Any,
        processed_payload: Dict[str, Any],
        pages: List[Dict[str, Any]],
        logger: logging.Logger,
    ) -> List[List[float]]:
        embeddings: List[List[float]] = []
        total = len(pages)
        for index, page in enumerate(pages, start=1):
            page_number = page.get("page_number", "unknown")
            logger.info("Embedding page %s (%d/%d)", page_number, index, total)
            embeddings.append(
                self._embed_page(gemini_client, processed_payload, page, logger)
            )
            if index < total:
                time.sleep(self._settings.gemini_embedding_delay_sec)

        return embeddings

    def _embed_page(
        self,
        gemini_client: Any,
        processed_payload: Dict[str, Any],
        page: Dict[str, Any],
        logger: logging.Logger,
    ) -> List[float]:
        page_number = page.get("page_number", "unknown")
        return embed_text_with_retries(
            gemini_client=gemini_client,
            model=self._settings.gemini_embedding_model,
            text=self._document_embedding_text(processed_payload, page),
            max_retries=self._settings.gemini_embedding_max_retries,
            retry_delay_sec=self._settings.gemini_retry_delay_sec,
            logger=logger,
            label=f"page {page_number}",
        )

    def _document_embedding_text(
        self,
        processed_payload: Dict[str, Any],
        page: Dict[str, Any],
    ) -> str:
        company_name = str(processed_payload.get("company_name") or "unknown company")
        year = str(processed_payload.get("year") or "unknown year")
        section = str(page.get("section") or "unknown")
        title = f"{company_name} annual report {year} {section}"
        content = str(page.get("text_for_embedding") or "")
        return f"title: {title} | text: {content}"

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

    def _build_points(
        self,
        processed_payload: Dict[str, Any],
        page_items: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> List[PointStruct]:
        top_level_payload = {
            "pdf_name": processed_payload.get("pdf_name", ""),
            "company_name": processed_payload.get("company_name", "unknown"),
            "year": processed_payload.get("year", "unknown"),
        }

        points: List[PointStruct] = []
        for page, embedding in zip(page_items, embeddings):
            page_number = int(page.get("page_number") or 0)
            payload = {
                **top_level_payload,
                "page_number": page_number,
                "section": page.get("section", "unknown"),
                "has_table": bool(page.get("has_table", False)),
                "text": page.get("text", ""),
                "text_for_embedding": page.get("text_for_embedding", ""),
                "tables": page.get("tables", []),
            }
            points.append(
                PointStruct(
                    id=page_number,
                    vector=embedding,
                    payload=payload,
                )
            )

        return points

    def _recreate_collection(
        self,
        qdrant: QdrantClient,
        collection_name: str,
        vector_size: int,
    ) -> None:
        if qdrant.collection_exists(collection_name):
            qdrant.delete_collection(collection_name)

        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=self._distance(),
            ),
        )

    def _distance(self) -> Distance:
        normalized = self._settings.qdrant_distance.strip().lower()
        if normalized == "cosine":
            return Distance.COSINE
        if normalized in {"dot", "dot_product"}:
            return Distance.DOT
        if normalized in {"euclid", "euclidean"}:
            return Distance.EUCLID
        raise ValueError(f"Unsupported Qdrant distance: {self._settings.qdrant_distance}")

    def _build_collection_name(self, pdf_name: str) -> str:
        stem = Path(pdf_name).with_suffix("").name
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
        if not slug:
            slug = "unknown_report"
        return f"report_{slug}"

    def _error_response(
        self,
        processed_file_path: str,
        exc: Exception,
        logger: logging.Logger,
    ) -> VectorIngestResponse:
        error = str(exc)
        log_json_artifact(
            logger,
            "ingestion_summary.json",
            {
                "processed_file_path": processed_file_path,
                "collection_name": "",
                "points_inserted": 0,
                "status": "error",
                "error": error,
                "errors": [error],
            },
        )
        return VectorIngestResponse(status="error", error=error, errors=[error])
