from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ...core.config import Settings
from ...core.logger import log_json_artifact
from ...schemas.vector import VectorIngestResponse
from ..common.gemini import get_gemini_client


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
            response = await asyncio.to_thread(
                self._ingest_processed_file_sync,
                processed_file_path,
                logger,
            )
            return response
        except Exception as exc:
            logger.error("Vector ingestion failed: %s", exc)
            log_json_artifact(
                logger,
                "ingestion_summary.json",
                {
                    "processed_file_path": processed_file_path,
                    "collection_name": "",
                    "points_inserted": 0,
                    "status": "error",
                    "error": str(exc),
                    "errors": [str(exc)],
                },
            )
            return VectorIngestResponse(
                status="error",
                error=str(exc),
                errors=[str(exc)],
            )

    def _ingest_processed_file_sync(
        self,
        processed_file_path: str,
        logger: logging.Logger,
    ) -> VectorIngestResponse:
        path = self._resolve_processed_path(processed_file_path)
        processed_payload = self._load_processed_payload(path)

        pdf_name = str(processed_payload.get("pdf_name") or path.stem)
        collection_name = self._build_collection_name(pdf_name)
        logger.info("Resolved processed JSON: %s", path)
        logger.info("Target Qdrant collection: %s", collection_name)

        page_items, skipped_pages = self._build_page_items(processed_payload)
        if not page_items:
            raise ValueError("No pages with non-empty text_for_embedding found")

        qdrant = self._connect_to_qdrant(logger)

        embeddings = self._embed_pages(page_items, logger)
        vector_size = len(embeddings[0])
        points = self._build_points(processed_payload, page_items, embeddings)

        self._recreate_collection(qdrant, collection_name, vector_size)
        qdrant.upsert(collection_name=collection_name, points=points)

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

        return VectorIngestResponse(
            collection_name=collection_name,
            points_inserted=len(points),
            status="success",
            errors=[],
        )

    def _resolve_processed_path(self, processed_file_path: str) -> Path:
        candidate = Path(processed_file_path)
        if not candidate.is_absolute():
            candidate = self._settings.repository_root / candidate
        candidate = candidate.resolve()

        if not candidate.exists():
            raise FileNotFoundError(f"Processed file not found: {candidate}")
        if not candidate.is_file():
            raise ValueError(f"Processed path must point to a file: {candidate}")
        return candidate

    def _load_processed_payload(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if not isinstance(payload, dict):
            raise ValueError("Processed file must contain a JSON object")
        if not isinstance(payload.get("pages"), list):
            raise ValueError("Processed file must contain a pages list")
        return payload

    def _build_page_items(
        self,
        processed_payload: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], List[int]]:
        page_items: List[Dict[str, Any]] = []
        skipped_pages: List[int] = []

        for page in processed_payload.get("pages", []):
            if not isinstance(page, dict):
                continue

            page_number = int(page.get("page_number") or 0)
            text_for_embedding = str(page.get("text_for_embedding") or "").strip()
            if not text_for_embedding:
                skipped_pages.append(page_number)
                continue

            page_items.append(page)

        return page_items, skipped_pages

    def _embed_pages(
        self,
        page_items: List[Dict[str, Any]],
        logger: logging.Logger,
    ) -> List[List[float]]:
        gemini_client = get_gemini_client()
        if gemini_client is None:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set for embeddings")

        embeddings: List[List[float]] = []
        total = len(page_items)
        for index, page in enumerate(page_items, start=1):
            page_number = page.get("page_number", "unknown")
            logger.info("Embedding page %s (%d/%d)", page_number, index, total)
            response = gemini_client.models.embed_content(
                model=self._settings.gemini_embedding_model,
                contents=self._document_embedding_text(page),
            )
            embeddings.append(self._extract_embedding(response))

        return embeddings

    def _document_embedding_text(self, page: Dict[str, Any]) -> str:
        title = self._document_title(page)
        content = str(page.get("text_for_embedding") or "")
        return f"title: {title} | text: {content}"

    def _document_title(self, page: Dict[str, Any]) -> str:
        section = str(page.get("section") or "unknown")
        page_number = page.get("page_number", "unknown")
        return f"{section} page {page_number}"

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

    def _extract_embedding(self, response: Any) -> List[float]:
        embeddings = getattr(response, "embeddings", None)
        if embeddings:
            values = getattr(embeddings[0], "values", None)
            if values:
                return [float(value) for value in values]

        embedding = getattr(response, "embedding", None)
        if embedding is not None:
            values = getattr(embedding, "values", None)
            if values:
                return [float(value) for value in values]

        raise ValueError("Gemini embedding response did not contain vector values")

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
            "auditor_opinion": processed_payload.get("auditor_opinion", "unknown"),
            "auditor_firm": processed_payload.get("auditor_firm", "unknown"),
            "audit_period": processed_payload.get("audit_period", "unknown"),
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
