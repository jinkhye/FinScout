from __future__ import annotations

import logging
from typing import List, Sequence

from google.genai import types
from pydantic import BaseModel, Field

from ...core.config import Settings
from ...core.logger import log_json_artifact, log_text_artifact
from ...schemas.query import QueryPlanResponse
from ...schemas.vector import SectionLabel, VectorQueryResult
from ..common.gemini import get_gemini_client
from ..common.prompts import build_retrieval_repair_prompt


SUPPORTED_SECTIONS: tuple[SectionLabel, ...] = (
    "company_overview",
    "mda",
    "auditor_report",
    "balance_sheet",
    "income_statement",
    "equitychange_statement",
    "cashflow_statement",
    "notes",
    "other",
)


class RetrievalRepairOutput(BaseModel):
    repaired_query: str = Field(..., min_length=1)
    selected_sections: List[SectionLabel] = Field(default_factory=list)
    reason: str = ""


class RetrievalRepairService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def repair(
        self,
        *,
        question: str,
        planner: QueryPlanResponse,
        current_query: str,
        current_sections: Sequence[SectionLabel],
        results: Sequence[VectorQueryResult],
        logger: logging.Logger,
    ) -> RetrievalRepairOutput:
        client = get_gemini_client()
        if client is None:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY must be set for retrieval repair"
            )

        prompt = build_retrieval_repair_prompt(
            question=question,
            planner=planner,
            current_query=current_query,
            current_sections=current_sections,
            results=results,
        )
        log_text_artifact(logger, "retry_prompt.md", prompt + "\n")

        response = client.models.generate_content(
            model=self._settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=RetrievalRepairOutput.model_json_schema(),
            ),
        )
        response_text = (getattr(response, "text", None) or "").strip()
        if not response_text:
            raise ValueError("Gemini returned an empty retrieval repair response")

        output = RetrievalRepairOutput.model_validate_json(response_text)
        normalized = self._normalize_output(output)
        log_json_artifact(logger, "retry_output.json", normalized.model_dump())
        return normalized

    def _normalize_output(
        self,
        output: RetrievalRepairOutput,
    ) -> RetrievalRepairOutput:
        repaired_query = output.repaired_query.strip()
        if not repaired_query:
            raise ValueError("Repaired retrieval query cannot be blank")

        selected_sections: List[SectionLabel] = []
        for section in output.selected_sections:
            if section in SUPPORTED_SECTIONS and section not in selected_sections:
                selected_sections.append(section)

        return RetrievalRepairOutput(
            repaired_query=repaired_query,
            selected_sections=selected_sections,
            reason=output.reason.strip(),
        )
