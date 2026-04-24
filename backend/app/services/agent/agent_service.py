from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
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
from ..common.prompts import build_answer_prompt, build_direct_reply_prompt
from ..query_planning.query_context_service import QueryContextService
from ..query_planning.query_planner_service import QueryPlannerService
from ..vector_ingestion.vector_index import (
    build_collection_name,
    load_processed_payload,
)
from .conversation_memory_service import ConversationMemoryService, ConversationTurn
from .retrieval_repair_service import RetrievalRepairService
from .reranker_service import RerankerService
from ..vector_ingestion.vector_query_service import VectorQueryService


class AnswerCitation(BaseModel):
    page_number: int
    section: str


class AnswerOutput(BaseModel):
    answer: str
    citations: List[AnswerCitation] = Field(default_factory=list)


@dataclass
class ContextBuildResult:
    context_text: str
    evidence: List[Dict[str, Any]]
    selected_sections: List[str]
    reranked: bool
    retried: bool
    final_query: str
    retry_reason: str | None


class AgentService:
    def __init__(
        self,
        settings: Settings,
        planner: QueryPlannerService,
        context_loader: QueryContextService,
        memory: ConversationMemoryService,
        repair: RetrievalRepairService,
        reranker: RerankerService,
        vector_query: VectorQueryService,
    ) -> None:
        self._settings = settings
        self._planner = planner
        self._context_loader = context_loader
        self._memory = memory
        self._repair = repair
        self._reranker = reranker
        self._vector_query = vector_query

    async def ask(
        self,
        request: AgentAskRequest,
        logger: logging.Logger,
    ) -> AgentAskResponse:
        try:
            processed_path, processed_payload = load_processed_payload(
                self._settings,
                request.processed_file_path,
            )
            collection_name = self._session_collection_name(
                request,
                processed_path=str(processed_path),
                processed_payload=processed_payload,
            )
            session = self._memory.get_or_create_session(
                session_id=request.session_id,
                processed_file_path=str(processed_path),
                collection_name=collection_name,
                company_name=str(processed_payload.get("company_name") or "unknown"),
                year=str(processed_payload.get("year") or "unknown"),
            )
            turn_index = self._memory.next_turn_index(request.session_id)
            recent_turns = self._memory.list_recent_turns(request.session_id, limit=3)
            planner_conversation_context = self._planner_conversation_context(
                recent_turns
            )
            answer_conversation_context = self._answer_conversation_context(
                recent_turns
            )
            self._log_conversation_context(
                logger=logger,
                session_id=session.session_id,
                turn_index=turn_index,
                recent_turns=recent_turns,
                planner_context=planner_conversation_context,
                answer_context=answer_conversation_context,
            )

            logger.info("Agent ask request for %s", request.processed_file_path)
            logger.info("Session ID: %s", request.session_id)
            logger.info("Turn index: %d", turn_index)
            logger.info(
                "Conversation history injected: %s",
                "yes" if recent_turns else "no",
            )
            logger.info("User question: %s", request.question)
            planner = await self._planner.plan(
                QueryPlanRequest(
                    processed_file_path=request.processed_file_path,
                    query=request.question,
                    conversation_context=planner_conversation_context,
                ),
                logger=logger,
            )
            if planner.status != "success":
                raise ValueError(planner.error or "Query planner failed")
            logger.info("Planner intent: %s", planner.intent)
            logger.info("Optimized query: %s", planner.optimized_query)

            if planner.intent == "direct_reply":
                logger.info("Retrieval skipped for direct-reply turn")
                prompt = build_direct_reply_prompt(
                    question=request.question,
                    company_name=planner.company_name,
                    year=planner.year,
                    conversation_context=answer_conversation_context,
                )
                evidence: List[Dict[str, Any]] = []
                reranked = False
                retried = False
                final_query = planner.optimized_query
                retry_reason = None
                selected_sections_for_memory: List[str] = []
            else:
                context_result = await self._build_context(
                    request,
                    planner,
                    collection_name,
                    logger,
                )
                context_text = context_result.context_text
                evidence = context_result.evidence
                reranked = context_result.reranked
                retried = context_result.retried
                final_query = context_result.final_query
                retry_reason = context_result.retry_reason
                selected_sections_for_memory = context_result.selected_sections
                prompt = build_answer_prompt(
                    question=request.question,
                    planner=planner,
                    context_text=context_text,
                    conversation_context=answer_conversation_context,
                )

            log_text_artifact(logger, "answer_prompt.md", prompt + "\n")
            answer_output = self._call_answer_model(prompt)

            response = AgentAskResponse(
                session_id=request.session_id,
                turn_index=turn_index,
                question=request.question,
                answer=answer_output.answer.strip(),
                company_name=planner.company_name,
                year=planner.year,
                route_strategy=planner.route_strategy,
                reranked=reranked,
                retried=retried,
                final_query=final_query,
                retry_reason=retry_reason,
                citations=(
                    []
                    if planner.intent == "direct_reply"
                    else self._normalize_citations(answer_output.citations, evidence)
                ),
                planner=planner,
                status="success",
                errors=[],
            )
            self._memory.append_turn(
                session_id=request.session_id,
                turn_index=turn_index,
                question=request.question,
                intent=planner.intent,
                optimized_query=final_query,
                selected_sections=selected_sections_for_memory,
                route_strategy=planner.route_strategy or "",
                reranked=reranked,
                answer=response.answer,
                citations=[citation.model_dump() for citation in response.citations],
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
        collection_name: str,
        logger: logging.Logger,
    ) -> ContextBuildResult:
        if planner.route_strategy == "full_context":
            context_response = await self._context_loader.load_context(
                QueryContextRequest(
                    processed_file_path=request.processed_file_path,
                    sections=planner.full_context_sections,
                ),
                logger=logger,
            )
            if context_response.status != "success":
                raise ValueError(
                    context_response.error or "Full-context loading failed"
                )

            log_json_artifact(
                logger, "context_output.json", context_response.model_dump()
            )
            return ContextBuildResult(
                context_text=context_response.context_text,
                evidence=self._evidence_from_context(
                    context_response.context_text,
                    (
                        planner.full_context_sections[0]
                        if planner.full_context_sections
                        else ""
                    ),
                ),
                selected_sections=list(planner.full_context_sections),
                reranked=False,
                retried=False,
                final_query=planner.optimized_query,
                retry_reason=None,
            )

        current_query = planner.optimized_query
        current_sections = list(planner.vector_search_sections)
        vector_response = await self._vector_query.query(
            self._vector_query_request(
                request=request,
                collection_name=collection_name,
                query=current_query,
                sections=current_sections,
            ),
            logger=logger,
        )
        if vector_response.status != "success":
            raise ValueError(vector_response.error or "Vector retrieval failed")

        self._log_retrieval_artifact(
            logger=logger,
            artifact_name="initial_retrieval_output.json",
            query=current_query,
            sections=current_sections,
            response=vector_response.model_dump(),
        )
        rerank_result = self._reranker.rerank(
            request=request,
            planner=planner,
            results=vector_response.results,
            logger=logger,
        )
        logger.info(
            "Initial retrieval judged %s",
            "weak" if rerank_result.retry_recommended else "strong",
        )
        if not rerank_result.retry_recommended:
            logger.info("Retry skipped because the first retrieval looked strong")

        final_results = rerank_result.results
        reranked = rerank_result.reranked
        retried = False
        retry_reason = None
        final_query = current_query
        final_sections = current_sections
        final_response_dump = vector_response.model_dump()

        if rerank_result.retry_recommended:
            logger.info(
                "Retry triggered after weak retrieval: %s",
                rerank_result.retry_reason or "no reason provided",
            )
            try:
                repair_output = self._repair.repair(
                    question=request.question,
                    planner=planner,
                    current_query=current_query,
                    current_sections=current_sections,
                    results=vector_response.results,
                    logger=logger,
                )
                logger.info("Repaired query: %s", repair_output.repaired_query)
                if repair_output.selected_sections != current_sections:
                    logger.info(
                        "Retry section filter changed from %s to %s",
                        current_sections or ["all"],
                        repair_output.selected_sections or ["all"],
                    )

                retry_response = await self._vector_query.query(
                    self._vector_query_request(
                        request=request,
                        collection_name=collection_name,
                        query=repair_output.repaired_query,
                        sections=repair_output.selected_sections,
                    ),
                    logger=logger,
                )
                if retry_response.status != "success":
                    raise ValueError(retry_response.error or "Retry retrieval failed")

                retry_rerank = self._reranker.rerank(
                    request=request,
                    planner=planner,
                    results=retry_response.results,
                    logger=logger,
                    artifact_prefix="retry_",
                )
                final_results = retry_rerank.results
                reranked = retry_rerank.reranked
                retried = True
                retry_reason = repair_output.reason or rerank_result.retry_reason
                final_query = repair_output.repaired_query
                final_sections = list(repair_output.selected_sections)
                final_response_dump = retry_response.model_dump()
            except Exception as exc:
                logger.warning(
                    "Retry failed, continuing with initial retrieval: %s", exc
                )

        self._log_retrieval_artifact(
            logger=logger,
            artifact_name="final_retrieval_output.json",
            query=final_query,
            sections=final_sections,
            response=final_response_dump,
        )

        return ContextBuildResult(
            context_text=self._context_from_results(final_results),
            evidence=self._evidence_from_results(final_results),
            selected_sections=final_sections,
            reranked=reranked,
            retried=retried,
            final_query=final_query,
            retry_reason=retry_reason,
        )

    def _vector_query_request(
        self,
        *,
        request: AgentAskRequest,
        collection_name: str,
        query: str,
        sections: List[str],
    ) -> VectorQueryRequest:
        filters = self._vector_filters(sections)
        return VectorQueryRequest(
            processed_file_path=request.processed_file_path,
            collection_name=collection_name,
            query=query,
            top_k=request.top_k,
            filters=filters,
        )

    def _vector_filters(self, sections: List[str]) -> VectorQueryFilters | None:
        if not sections:
            return None
        return VectorQueryFilters(sections=sections)

    def _log_retrieval_artifact(
        self,
        *,
        logger: logging.Logger,
        artifact_name: str,
        query: str,
        sections: List[str],
        response: Dict[str, Any],
    ) -> None:
        log_json_artifact(
            logger,
            artifact_name,
            {
                "query": query,
                "selected_sections": sections,
                "response": response,
            },
        )

    def _call_answer_model(self, prompt: str) -> AnswerOutput:
        client = get_gemini_client()
        if client is None:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY must be set for answering"
            )

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
            parts.append(
                f"<!-- Page {result.page_number} | section={result.section} -->"
            )
            parts.append("")
            parts.append(result.text)
            parts.append("")
            parts.append("---")
            parts.append("")
        return "\n".join(parts).strip() + "\n"

    def _evidence_from_results(
        self, results: List[VectorQueryResult]
    ) -> List[Dict[str, Any]]:
        return [
            {
                "page_number": result.page_number,
                "section": result.section,
                "excerpt": self._citation_excerpt(result.text),
            }
            for result in results
            if result.page_number is not None
        ]

    def _evidence_from_context(
        self, context_text: str, section: str
    ) -> List[Dict[str, Any]]:
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
            session_id=request.session_id,
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

    def _session_collection_name(
        self,
        request: AgentAskRequest,
        *,
        processed_path: str,
        processed_payload: Dict[str, Any],
    ) -> str:
        if request.collection_name and request.collection_name.strip():
            return request.collection_name.strip()
        pdf_name = str(processed_payload.get("pdf_name") or Path(processed_path).stem)
        return build_collection_name(pdf_name)

    def _planner_conversation_context(self, turns: List[ConversationTurn]) -> str:
        if not turns:
            return ""

        parts: List[str] = []
        for turn in turns:
            parts.append(f"Turn {turn.turn_index}")
            parts.append(f"User question: {turn.question}")
            parts.append(
                f"Assistant answer summary: {self._truncate(turn.answer, 220)}"
            )
            parts.append("")
        return "\n".join(parts).strip()

    def _answer_conversation_context(self, turns: List[ConversationTurn]) -> str:
        if not turns:
            return ""

        parts: List[str] = []
        for turn in turns:
            citations = ", ".join(
                f"p.{citation.get('page_number')} ({citation.get('section')})"
                for citation in turn.citations
                if citation.get("page_number") is not None
            )
            parts.append(f"Turn {turn.turn_index}")
            parts.append(f"User question: {turn.question}")
            parts.append(
                f"Assistant answer summary: {self._truncate(turn.answer, 260)}"
            )
            if citations:
                parts.append(f"Citations: {citations}")
            parts.append("")
        return "\n".join(parts).strip()

    def _log_conversation_context(
        self,
        *,
        logger: logging.Logger,
        session_id: str,
        turn_index: int,
        recent_turns: List[ConversationTurn],
        planner_context: str,
        answer_context: str,
    ) -> None:
        log_json_artifact(
            logger,
            "conversation_context.json",
            {
                "session_id": session_id,
                "turn_index": turn_index,
                "recent_turns_count": len(recent_turns),
                "recent_turns": [
                    {
                        "turn_index": turn.turn_index,
                        "question": turn.question,
                        "intent": turn.intent,
                        "optimized_query": turn.optimized_query,
                        "selected_sections": turn.selected_sections,
                        "route_strategy": turn.route_strategy,
                        "reranked": turn.reranked,
                        "answer_summary": self._truncate(turn.answer, 260),
                        "citations": turn.citations,
                        "created_at": turn.created_at,
                    }
                    for turn in recent_turns
                ],
                "planner_context": planner_context,
                "answer_context": answer_context,
            },
        )

    def _truncate(self, text: str, limit: int) -> str:
        normalized = " ".join(str(text).split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."
