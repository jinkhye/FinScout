from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from ...core.config import Settings
from ...core.logger import log_json_artifact
from ...schemas.query import QueryContextRequest, QueryContextResponse
from ...schemas.vector import SectionLabel
from .query_planner_service import SHORT_FULL_CONTEXT_SECTIONS


class QueryContextService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def load_context(
        self,
        request: QueryContextRequest,
        logger: logging.Logger,
    ) -> QueryContextResponse:
        try:
            logger.info("Query context request for %s", request.processed_file_path)
            return await asyncio.to_thread(self._load_context_sync, request, logger)
        except Exception as exc:
            logger.error("Query context failed: %s", exc)
            return self._error_response(request, exc, logger)

    def _load_context_sync(
        self,
        request: QueryContextRequest,
        logger: logging.Logger,
    ) -> QueryContextResponse:
        self._validate_sections(request.sections)
        section = request.sections[0]

        path, processed_payload = self._load_processed_payload(
            request.processed_file_path
        )
        pages = self._section_pages(processed_payload, section)
        if not pages:
            raise ValueError(f"No pages found for section: {section}")

        context_text = self._build_context_text(pages)
        response = QueryContextResponse(
            processed_file_path=str(path),
            company_name=self._metadata_value(processed_payload, "company_name"),
            year=self._metadata_value(processed_payload, "year"),
            sections=[section],
            pages_count=len(pages),
            context_text=context_text,
            status="success",
            errors=[],
        )
        log_json_artifact(logger, "context_output.json", response.model_dump())
        logger.info(
            "Loaded %d pages for full-context section %s",
            response.pages_count,
            section,
        )
        return response

    def _validate_sections(self, sections: List[SectionLabel]) -> None:
        if not sections:
            raise ValueError("Exactly one section is required")
        if len(sections) != 1:
            raise ValueError("Full-context retrieval only supports exactly one section")

        section = sections[0]
        if section not in SHORT_FULL_CONTEXT_SECTIONS:
            allowed = ", ".join(sorted(SHORT_FULL_CONTEXT_SECTIONS))
            raise ValueError(
                f"Section '{section}' is not eligible for full-context retrieval. "
                f"Allowed sections: {allowed}"
            )

    def _load_processed_payload(
        self,
        processed_file_path: str,
    ) -> tuple[Path, Dict[str, Any]]:
        candidate = Path(processed_file_path)
        if not candidate.is_absolute():
            candidate = self._settings.repository_root / candidate
        candidate = candidate.resolve()

        if not candidate.exists():
            raise FileNotFoundError(f"Processed file not found: {candidate}")
        if not candidate.is_file():
            raise ValueError(f"Processed path must point to a file: {candidate}")

        with candidate.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if not isinstance(payload, dict):
            raise ValueError("Processed file must contain a JSON object")
        if not isinstance(payload.get("pages"), list):
            raise ValueError("Processed file must contain a pages list")
        return candidate, payload

    def _section_pages(
        self,
        processed_payload: Dict[str, Any],
        section: SectionLabel,
    ) -> List[Dict[str, Any]]:
        pages: List[Dict[str, Any]] = []
        for page in processed_payload.get("pages", []):
            if not isinstance(page, dict):
                continue
            if page.get("section") == section:
                pages.append(page)

        pages.sort(key=lambda page: int(page.get("page_number") or 0))
        return pages

    def _build_context_text(self, pages: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for page in pages:
            page_number = page.get("page_number", "unknown")
            section = page.get("section", "unknown")
            parts.append(f"<!-- Page {page_number} | section={section} -->")
            parts.append("")
            parts.append(str(page.get("text") or ""))
            parts.append("")
            parts.append("---")
            parts.append("")

        return "\n".join(parts).strip() + "\n"

    def _metadata_value(self, processed_payload: Dict[str, Any], key: str) -> str:
        value = processed_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return "unknown"

    def _error_response(
        self,
        request: QueryContextRequest,
        exc: Exception,
        logger: logging.Logger,
    ) -> QueryContextResponse:
        error = str(exc)
        log_json_artifact(
            logger,
            "context_output.json",
            {
                "processed_file_path": request.processed_file_path,
                "sections": request.sections,
                "pages_count": 0,
                "status": "error",
                "error": error,
                "errors": [error],
            },
        )
        return QueryContextResponse(
            processed_file_path=request.processed_file_path,
            sections=request.sections,
            status="error",
            error=error,
            errors=[error],
        )
