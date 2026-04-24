from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from google.genai import types

from ...core.config import Settings
from ...core.logger import log_json_artifact, log_text_artifact
from ...schemas.query import (
    QueryPlanRequest,
    QueryPlanResponse,
    QueryPlannerModelOutput,
)
from ...schemas.vector import SectionLabel
from ..common.gemini import get_gemini_client
from ..common.prompts import build_query_planner_prompt


SUPPORTED_SECTIONS: List[SectionLabel] = [
    "company_overview",
    "mda",
    "auditor_report",
    "balance_sheet",
    "income_statement",
    "equitychange_statement",
    "cashflow_statement",
    "notes",
    "other",
]

SHORT_FULL_CONTEXT_SECTIONS: set[SectionLabel] = {
    "auditor_report",
    "balance_sheet",
    "income_statement",
    "equitychange_statement",
    "cashflow_statement",
}


class QueryPlannerService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def plan(
        self,
        request: QueryPlanRequest,
        logger: logging.Logger,
    ) -> QueryPlanResponse:
        try:
            logger.info("Query planning request for %s", request.processed_file_path)
            return await asyncio.to_thread(self._plan_sync, request, logger)
        except Exception as exc:
            logger.error("Query planning failed: %s", exc)
            return self._error_response(request, exc, logger)

    def _plan_sync(
        self,
        request: QueryPlanRequest,
        logger: logging.Logger,
    ) -> QueryPlanResponse:
        processed_payload = self._load_processed_payload(request.processed_file_path)
        company_name = self._metadata_value(processed_payload, "company_name")
        year = self._metadata_value(processed_payload, "year")
        available_sections = self._available_sections(processed_payload)

        client = get_gemini_client()
        if client is None:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY must be set for planning"
            )

        prompt = build_query_planner_prompt(
            query=request.query,
            company_name=company_name,
            year=year,
            available_sections=available_sections,
            conversation_context=request.conversation_context,
        )
        log_text_artifact(logger, "planner_prompt.md", prompt + "\n")

        model_output = self._call_planner_model(client, prompt)
        normalized_output = self._normalize_model_output(model_output)
        route_strategy, vector_sections, full_context_sections = self._route_sections(
            normalized_output.intent,
            normalized_output.selected_sections,
        )

        response = QueryPlanResponse(
            original_query=request.query,
            intent=normalized_output.intent,
            optimized_query=normalized_output.optimized_query,
            company_name=company_name,
            year=year,
            selected_sections=normalized_output.selected_sections,
            route_strategy=route_strategy,
            vector_search_sections=vector_sections,
            full_context_sections=full_context_sections,
            status="success",
            errors=[],
        )
        log_json_artifact(logger, "planner_output.json", response.model_dump())
        logger.info(
            "Query plan completed with intent=%s route_strategy=%s sections=%s",
            response.intent,
            response.route_strategy,
            response.selected_sections,
        )
        return response

    def _load_processed_payload(self, processed_file_path: str) -> Dict[str, Any]:
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
        return payload

    def _metadata_value(self, processed_payload: Dict[str, Any], key: str) -> str:
        value = processed_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return "unknown"

    def _available_sections(self, processed_payload: Dict[str, Any]) -> List[str]:
        classified_pages = processed_payload.get("classified_pages")
        if not isinstance(classified_pages, dict):
            return list(SUPPORTED_SECTIONS)

        sections = [
            section
            for section in SUPPORTED_SECTIONS
            if section in classified_pages and classified_pages.get(section)
        ]
        return sections or list(SUPPORTED_SECTIONS)

    def _call_planner_model(
        self,
        client: Any,
        prompt: str,
    ) -> QueryPlannerModelOutput:
        response = client.models.generate_content(
            model=self._settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=QueryPlannerModelOutput.model_json_schema(),
            ),
        )
        response_text = (getattr(response, "text", None) or "").strip()
        if not response_text:
            raise ValueError("Gemini returned an empty query planner response")
        return QueryPlannerModelOutput.model_validate_json(response_text)

    def _normalize_model_output(
        self,
        model_output: QueryPlannerModelOutput,
    ) -> QueryPlannerModelOutput:
        sections: List[SectionLabel] = []
        for section in model_output.selected_sections:
            if section in SUPPORTED_SECTIONS and section not in sections:
                sections.append(section)

        return QueryPlannerModelOutput(
            intent=model_output.intent,
            optimized_query=model_output.optimized_query.strip(),
            selected_sections=[] if model_output.intent == "direct_reply" else sections,
        )

    def _route_sections(
        self,
        intent: str,
        selected_sections: List[SectionLabel],
    ) -> tuple[str | None, List[SectionLabel], List[SectionLabel]]:
        if intent == "direct_reply":
            return None, [], []
        if not selected_sections:
            return "vector_search", [], []

        if (
            len(selected_sections) == 1
            and selected_sections[0] in SHORT_FULL_CONTEXT_SECTIONS
        ):
            return "full_context", [], selected_sections

        return "vector_search", selected_sections, []

    def _error_response(
        self,
        request: QueryPlanRequest,
        exc: Exception,
        logger: logging.Logger,
    ) -> QueryPlanResponse:
        error = str(exc)
        log_json_artifact(
            logger,
            "planner_output.json",
            {
                "original_query": request.query,
                "intent": "report_question",
                "optimized_query": "",
                "status": "error",
                "error": error,
                "errors": [error],
            },
        )
        return QueryPlanResponse(
            original_query=request.query,
            status="error",
            error=error,
            errors=[error],
        )
