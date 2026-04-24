from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from google.genai import types
from pydantic import BaseModel, Field

from ...core.config import Settings
from ...core.logger import log_json_artifact, log_text_artifact
from ...schemas.agent import AgentAskRequest, AgentAskResponse, AgentCitation
from ...schemas.query import QueryContextRequest, QueryPlanRequest, QueryPlanResponse
from ...schemas.vector import (
    VectorQueryFilters,
    VectorQueryRequest,
    VectorQueryResult,
)
from ..common.gemini import get_gemini_client
from ..common.prompts import build_answer_prompt, build_reranker_prompt
from ..query_planning.query_context_service import QueryContextService
from ..query_planning.query_planner_service import QueryPlannerService
from ..vector_ingestion.vector_index import resolve_collection_name
from ..vector_ingestion.vector_query_service import VectorQueryService


class RerankSelection(BaseModel):
    rank: int = Field(..., ge=1)
    page_number: int
    section: str
    reason: str = ""


class RerankerOutput(BaseModel):
    selected_results: List[RerankSelection] = Field(default_factory=list)


class AnswerCitation(BaseModel):
    page_number: int
    section: str


class AnswerOutput(BaseModel):
    answer: str = Field(...)
    citations: List[AnswerCitation] = Field(default_factory=list)


class AgentService:
    def __init__(
        self,
        settings: Settings,
        planner: QueryPlannerService,
        context_loader: QueryContextService,
        vector_query: VectorQueryService,
    ) -> None:
        self._settings = settings
        self._planner = planner
        self._context_loader = context_loader
        self._vector_query = vector_query

    async def ask(
        self,
        request: AgentAskRequest,
        logger: logging.Logger,
    ) -> AgentAskResponse:
        try:
            logger.info("Agent ask request for %s", request.processed_file_path)
            logger.info("User question: %s", request.question)
            planner = await self._planner.plan(
                QueryPlanRequest(
                    processed_file_path=request.processed_file_path,
                    query=request.question,
                ),
                logger=logger,
            )
            if planner.status != "success":
                raise ValueError(planner.error or "Query planner failed")
            logger.info("Optimized query: %s", planner.optimized_query)

            context_text, evidence = await self._build_context(
                request,
                planner,
                logger,
            )
            prompt = build_answer_prompt(
                question=request.question,
                planner=planner,
                context_text=context_text,
            )
            log_text_artifact(logger, "answer_prompt.md", prompt + "\n")
            answer_output = self._call_answer_model(prompt)

            response = AgentAskResponse(
                question=request.question,
                answer=answer_output.answer.strip(),
                company_name=planner.company_name,
                year=planner.year,
                route_strategy=planner.route_strategy,
                citations=self._normalize_citations(answer_output.citations, evidence),
                planner=planner,
                status="success",
                errors=[],
            )
            log_json_artifact(logger, "answer_output.json", response.model_dump())
            logger.info("Answer preview: %s", self._answer_preview(response.answer))
            logger.info("Agent answer completed successfully")
            return response
        except Exception as exc:
            logger.error("Agent ask failed: %s", exc)
            return self._error_response(request, exc, logger)

    async def _build_context(
        self,
        request: AgentAskRequest,
        planner: QueryPlanResponse,
        logger: logging.Logger,
    ) -> tuple[str, List[Dict[str, Any]]]:
        if planner.route_strategy == "full_context":
            context_response = await self._context_loader.load_context(
                QueryContextRequest(
                    processed_file_path=request.processed_file_path,
                    sections=planner.full_context_sections,
                ),
                logger=logger,
            )
            if context_response.status != "success":
                raise ValueError(context_response.error or "Full-context loading failed")

            log_json_artifact(logger, "context_output.json", context_response.model_dump())
            return context_response.context_text, self._evidence_from_context(
                context_response.context_text,
                planner.full_context_sections[0] if planner.full_context_sections else "",
            )

        collection_name = resolve_collection_name(
            self._settings,
            request.processed_file_path,
            request.collection_name,
        )
        vector_response = await self._vector_query.query(
            VectorQueryRequest(
                processed_file_path=request.processed_file_path,
                collection_name=collection_name,
                query=planner.optimized_query,
                top_k=request.top_k,
                filters=self._vector_filters(planner),
            ),
            logger=logger,
        )
        if vector_response.status != "success":
            raise ValueError(vector_response.error or "Vector retrieval failed")

        log_json_artifact(logger, "retrieval_output.json", vector_response.model_dump())
        results = vector_response.results
        if request.rerank:
            results = self._rerank_results(request, planner, results, logger)

        return self._context_from_results(results), self._evidence_from_results(results)

    def _vector_filters(self, planner: QueryPlanResponse) -> VectorQueryFilters | None:
        if planner.no_filter or not planner.vector_search_sections:
            return None
        return VectorQueryFilters(sections=planner.vector_search_sections)

    def _rerank_results(
        self,
        request: AgentAskRequest,
        planner: QueryPlanResponse,
        results: List[VectorQueryResult],
        logger: logging.Logger,
    ) -> List[VectorQueryResult]:
        if not results:
            return results

        try:
            prompt = build_reranker_prompt(
                question=request.question,
                planner=planner,
                results=results,
            )
            log_text_artifact(logger, "reranker_prompt.md", prompt + "\n")
            reranker_output = self._call_reranker_model(prompt)
            log_json_artifact(logger, "reranker_output.json", reranker_output.model_dump())

            by_page = {result.page_number: result for result in results}
            reranked: List[VectorQueryResult] = []
            for selection in sorted(reranker_output.selected_results, key=lambda item: item.rank):
                result = by_page.get(selection.page_number)
                if result is not None and result not in reranked:
                    reranked.append(result)

            return reranked or results[:5]
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
            return results

    def _call_reranker_model(self, prompt: str) -> RerankerOutput:
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

    def _call_answer_model(self, prompt: str) -> AnswerOutput:
        client = get_gemini_client()
        if client is None:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set for answering")

        response = client.models.generate_content(
            model=self._settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=AnswerOutput.model_json_schema(),
            ),
        )
        response_text = (getattr(response, "text", None) or "").strip()
        if not response_text:
            raise ValueError("Gemini returned an empty answer response")
        return AnswerOutput.model_validate_json(response_text)

    def _context_from_results(self, results: List[VectorQueryResult]) -> str:
        parts: List[str] = []
        for result in results:
            parts.append(f"<!-- Page {result.page_number} | section={result.section} -->")
            parts.append("")
            parts.append(result.text)
            parts.append("")
            parts.append("---")
            parts.append("")
        return "\n".join(parts).strip() + "\n"

    def _evidence_from_results(self, results: List[VectorQueryResult]) -> List[Dict[str, Any]]:
        return [
            {
                "page_number": result.page_number,
                "section": result.section,
                "excerpt": self._citation_excerpt(result.text),
            }
            for result in results
            if result.page_number is not None
        ]

    def _evidence_from_context(self, context_text: str, section: str) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        parts = [part.strip() for part in context_text.split("\n---\n") if part.strip()]
        for part in parts:
            lines = part.splitlines()
            if not lines or not lines[0].startswith("<!-- Page "):
                continue
            page_part = lines[0].removeprefix("<!-- Page ").split("|", 1)[0].strip()
            try:
                page_number = int(page_part)
            except ValueError:
                continue
            body = "\n".join(lines[1:]).strip()
            evidence.append(
                {
                    "page_number": page_number,
                    "section": section,
                    "excerpt": self._citation_excerpt(body),
                }
            )
        return evidence

    def _normalize_citations(
        self,
        citations: List[AnswerCitation],
        evidence: List[Dict[str, Any]],
    ) -> List[AgentCitation]:
        evidence_by_page = {
            int(item["page_number"]): {
                "section": str(item["section"]),
                "excerpt": str(item.get("excerpt") or ""),
            }
            for item in evidence
            if item.get("page_number") is not None
        }
        normalized: List[AgentCitation] = []
        for citation in citations:
            metadata = evidence_by_page.get(
                citation.page_number,
                {"section": citation.section, "excerpt": ""},
            )
            item = AgentCitation(
                page_number=citation.page_number,
                section=str(metadata["section"]),
                excerpt=str(metadata["excerpt"]),
            )
            if item not in normalized:
                normalized.append(item)

        if normalized:
            return normalized

        return [
            AgentCitation(
                page_number=page_number,
                section=str(metadata["section"]),
                excerpt=str(metadata["excerpt"]),
            )
            for page_number, metadata in evidence_by_page.items()
        ]

    def _error_response(
        self,
        request: AgentAskRequest,
        exc: Exception,
        logger: logging.Logger,
    ) -> AgentAskResponse:
        error = str(exc)
        log_json_artifact(
            logger,
            "answer_output.json",
            {
                "question": request.question,
                "answer": "",
                "status": "error",
                "error": error,
                "errors": [error],
            },
        )
        return AgentAskResponse(
            question=request.question,
            status="error",
            error=error,
            errors=[error],
        )

    def _answer_preview(self, answer: str, limit: int = 240) -> str:
        preview = " ".join(answer.split())
        if len(preview) <= limit:
            return preview
        return preview[: limit - 3].rstrip() + "..."

    def _citation_excerpt(self, text: str, limit: int = 280) -> str:
        excerpt = str(text)
        excerpt = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", excerpt)
        excerpt = re.sub(r"<!--.*?-->", " ", excerpt, flags=re.DOTALL)
        excerpt = re.sub(r"<[^>]+>", " ", excerpt)
        excerpt = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", excerpt)
        excerpt = re.sub(r"(\*\*|__|\*|_|~~)", "", excerpt)
        excerpt = re.sub(r"\s+", " ", excerpt).strip()
        if len(excerpt) <= limit:
            return excerpt
        return excerpt[: limit - 3].rstrip() + "..."
