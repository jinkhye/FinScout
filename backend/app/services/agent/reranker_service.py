from __future__ import annotations

import logging
from typing import List

from google.genai import types
from pydantic import BaseModel, Field

from ...core.config import Settings
from ...core.logger import log_json_artifact, log_text_artifact
from ...schemas.agent import AgentAskRequest
from ...schemas.query import QueryPlanResponse
from ...schemas.vector import VectorQueryResult
from ..common.gemini import get_gemini_client
from ..common.prompts import build_reranker_prompt


class RerankSelection(BaseModel):
    rank: int = Field(..., ge=1)
    page_number: int
    section: str
    reason: str = ""


class RerankerOutput(BaseModel):
    selected_results: List[RerankSelection] = Field(default_factory=list)


class RerankerService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def rerank(
        self,
        request: AgentAskRequest,
        planner: QueryPlanResponse,
        results: List[VectorQueryResult],
        logger: logging.Logger,
    ) -> tuple[List[VectorQueryResult], bool]:
        if not results:
            return results, False

        try:
            logger.info("Reranker started with %d candidates", len(results))
            prompt = build_reranker_prompt(
                question=request.question,
                planner=planner,
                results=results,
            )
            log_text_artifact(logger, "reranker_prompt.md", prompt + "\n")
            reranker_output = self._call_model(prompt)
            log_json_artifact(logger, "reranker_output.json", reranker_output.model_dump())

            by_page = {result.page_number: result for result in results}
            reranked: List[VectorQueryResult] = []
            for selection in sorted(reranker_output.selected_results, key=lambda item: item.rank):
                result = by_page.get(selection.page_number)
                if result is not None and result not in reranked:
                    reranked.append(result)

            selected_pages = [result.page_number for result in reranked]
            logger.info("Reranker selected pages: %s", selected_pages)
            if reranked:
                return reranked, True
            return results[:5], False
        except Exception as exc:
            logger.warning("Reranker failed, falling back to vector order: %s", exc)
            log_json_artifact(
                logger,
                "reranker_output.json",
                {
                    "status": "error",
                    "error": str(exc),
                    "fallback": "vector_order",
                },
            )
            return results, False

    def _call_model(self, prompt: str) -> RerankerOutput:
        client = get_gemini_client()
        if client is None:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set for reranking")

        response = client.models.generate_content(
            model=self._settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=RerankerOutput.model_json_schema(),
            ),
        )
        response_text = (getattr(response, "text", None) or "").strip()
        if not response_text:
            raise ValueError("Gemini returned an empty reranker response")
        return RerankerOutput.model_validate_json(response_text)
