from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, List, Literal

from fastapi.testclient import TestClient
from google.genai import types
from pydantic import BaseModel, Field

from ..core.config import get_settings
from ..core.logger import create_run_logger, log_json_artifact, log_text_artifact
from ..main import create_app
from ..services.common.gemini import get_gemini_client
from ..services.common.prompts import build_agent_ask_judge_prompt


JudgeBand = Literal["pass", "partial", "fail"]
HallucinationBand = Literal["none", "minor", "major"]


class GoldenEvalCase(BaseModel):
    id: str
    question: str
    expected_route_strategy: str | None = None
    expected_pages: List[int] = Field(default_factory=list)
    expected_facts: List[str] = Field(default_factory=list)
    notes: str = ""


class GoldenEvalDataset(BaseModel):
    processed_file_path: str
    cases: List[GoldenEvalCase]


class AgentAskJudgeOutput(BaseModel):
    passed: bool = Field(alias="pass")
    score: int = Field(..., ge=0, le=10)
    correctness: JudgeBand
    grounding: JudgeBand
    citation_quality: JudgeBand
    route_quality: JudgeBand
    hallucination: HallucinationBand
    reason: str = ""
    failures: List[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class EvalCaseResult(BaseModel):
    id: str
    question: str
    success: bool
    response_status_code: int
    answer_status: str
    answer: str = ""
    route_strategy: str | None = None
    citations: List[dict[str, Any]] = Field(default_factory=list)
    executed_steps: List[dict[str, Any]] = Field(default_factory=list)
    judge: AgentAskJudgeOutput | None = None
    error: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run golden evaluation for /agent/ask")
    parser.add_argument(
        "--dataset",
        default="backend/examples/agent_ask_golden_dataset.json",
        help="Path to golden dataset JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of cases to run",
    )
    return parser.parse_args()


def _load_dataset(dataset_path: str) -> GoldenEvalDataset:
    settings = get_settings()
    candidate = Path(dataset_path)
    if not candidate.is_absolute():
        candidate = settings.repository_root / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Dataset file not found: {candidate}")
    return GoldenEvalDataset.model_validate_json(candidate.read_text(encoding="utf-8"))


def _judge_case(case: GoldenEvalCase, response_payload: dict[str, Any]) -> AgentAskJudgeOutput:
    client = get_gemini_client()
    if client is None:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set for golden eval judging")

    prompt = build_agent_ask_judge_prompt(
        question=case.question,
        expected_route_strategy=case.expected_route_strategy,
        expected_pages=case.expected_pages,
        expected_facts=case.expected_facts,
        notes=case.notes,
        answer=str(response_payload.get("answer") or ""),
        citations=list(response_payload.get("citations") or []),
        route_strategy=response_payload.get("route_strategy"),
        executed_steps=list(response_payload.get("executed_steps") or []),
    )
    response = client.models.generate_content(
        model=get_settings().gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=AgentAskJudgeOutput.model_json_schema(),
        ),
    )
    response_text = (getattr(response, "text", None) or "").strip()
    if not response_text:
        raise ValueError("Gemini judge returned an empty response")
    return AgentAskJudgeOutput.model_validate_json(response_text)


def _summary_markdown(results: List[EvalCaseResult]) -> str:
    total = len(results)
    judged = [result for result in results if result.judge is not None]
    passed = sum(1 for result in judged if result.judge and result.judge.passed)
    average_score = (
        sum(result.judge.score for result in judged if result.judge is not None) / len(judged)
        if judged
        else 0.0
    )
    failure_counter = Counter()
    for result in judged:
        if result.judge is None:
            continue
        for failure in result.judge.failures:
            failure_counter[failure] += 1

    lines = [
        "# Agent Ask Golden Eval Summary",
        "",
        f"- Total cases: {total}",
        f"- Judged cases: {len(judged)}",
        f"- Passed: {passed}",
        f"- Pass rate: {(passed / len(judged) * 100):.1f}%" if judged else "- Pass rate: 0.0%",
        f"- Average score: {average_score:.2f}",
        "",
        "## Failure Breakdown",
    ]
    if failure_counter:
        lines.extend(f"- {name}: {count}" for name, count in failure_counter.most_common())
    else:
        lines.append("- No failure categories recorded.")

    lines.extend(["", "## Per-Case Results"])
    for result in results:
        if result.judge is None:
            lines.append(f"- {result.id}: ERROR - {result.error or 'No judge output'}")
            continue
        verdict = "PASS" if result.judge.passed else "FAIL"
        lines.append(
            f"- {result.id}: {verdict} | score={result.judge.score} | route={result.route_strategy} | {result.judge.reason}"
        )
    return "\n".join(lines) + "\n"


def run() -> int:
    args = _parse_args()
    dataset = _load_dataset(args.dataset)
    cases = dataset.cases[: args.limit] if args.limit else dataset.cases
    settings = get_settings()
    logger = create_run_logger(
        settings,
        "agent_ask_eval",
        {
            "dataset": args.dataset,
            "limit": args.limit,
            "cases": [case.id for case in cases],
        },
    )
    log_json_artifact(logger, "dataset_snapshot.json", dataset.model_dump())

    app = create_app()
    results: List[EvalCaseResult] = []
    run_id = getattr(logger, "run_dir").name

    with TestClient(app) as client:
        for case in cases:
            request_payload = {
                "session_id": f"golden-eval-{run_id}-{case.id}",
                "processed_file_path": dataset.processed_file_path,
                "question": case.question,
                "top_k": 8,
            }
            logger.info("Evaluating case %s", case.id)
            response = client.post("/api/v1/agent/ask", json=request_payload)
            try:
                response_payload = response.json()
            except Exception:
                response_payload = {}

            result = EvalCaseResult(
                id=case.id,
                question=case.question,
                success=response.status_code == 200 and response_payload.get("status") == "success",
                response_status_code=response.status_code,
                answer_status=str(response_payload.get("status") or "error"),
                answer=str(response_payload.get("answer") or ""),
                route_strategy=response_payload.get("route_strategy"),
                citations=list(response_payload.get("citations") or []),
                executed_steps=list(response_payload.get("executed_steps") or []),
                error=None if response.status_code == 200 else str(response_payload),
            )

            if result.success:
                judge_prompt = build_agent_ask_judge_prompt(
                    question=case.question,
                    expected_route_strategy=case.expected_route_strategy,
                    expected_pages=case.expected_pages,
                    expected_facts=case.expected_facts,
                    notes=case.notes,
                    answer=result.answer,
                    citations=result.citations,
                    route_strategy=result.route_strategy,
                    executed_steps=result.executed_steps,
                )
                log_text_artifact(logger, f"cases/{case.id}/judge_prompt.md", judge_prompt + "\n")
                try:
                    result.judge = _judge_case(case, response_payload)
                except Exception as exc:
                    result.error = str(exc)
            log_json_artifact(logger, f"cases/{case.id}/result.json", result.model_dump(by_alias=True))
            results.append(result)

    summary_payload = {
        "total_cases": len(results),
        "judged_cases": sum(1 for result in results if result.judge is not None),
        "passed_cases": sum(
            1 for result in results if result.judge is not None and result.judge.passed
        ),
        "results": [result.model_dump(by_alias=True) for result in results],
    }
    log_json_artifact(logger, "summary.json", summary_payload)
    log_text_artifact(logger, "summary.md", _summary_markdown(results))
    logger.info("Golden eval completed for %d cases", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
