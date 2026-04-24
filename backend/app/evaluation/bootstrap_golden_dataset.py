from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..core.config import get_settings
from ..core.logger import create_run_logger, log_json_artifact


DEFAULT_PROCESSED_FILE_PATH = (
    "backend/storage/pipelines/99SMART-Annual-Report-2024/processed_99SMART-Annual-Report-2024.json"
)


class SeedQuestion(BaseModel):
    id: str
    question: str
    notes: str = ""


class SeedQuestionSet(BaseModel):
    processed_file_path: str = DEFAULT_PROCESSED_FILE_PATH
    questions: list[SeedQuestion]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap a draft golden dataset by calling a live /api/v1/agent/ask endpoint"
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running backend",
    )
    parser.add_argument(
        "--questions",
        default="backend/examples/agent_ask_seed_questions.json",
        help="Path to seed questions JSON",
    )
    parser.add_argument(
        "--output",
        default="backend/examples/agent_ask_golden_dataset.draft.json",
        help="Path to write the draft golden dataset JSON",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="top_k value to send to /agent/ask",
    )
    return parser.parse_args()


def _resolve_repo_path(path_str: str) -> Path:
    settings = get_settings()
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = settings.repository_root / candidate
    return candidate.resolve()


def _load_seed_questions(path_str: str) -> SeedQuestionSet:
    path = _resolve_repo_path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Seed questions file not found: {path}")
    return SeedQuestionSet.model_validate_json(path.read_text(encoding="utf-8"))


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return slug or "case"


def _extract_expected_facts(answer: str) -> list[str]:
    cleaned = " ".join(answer.split())
    if not cleaned:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    facts: list[str] = []
    for sentence in sentences:
        compact = sentence.strip()
        if not compact:
            continue
        facts.append(compact)
        if len(facts) >= 3:
            break
    return facts


def _post_json(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            payload = json.loads(body)
        except Exception:
            payload = {"raw": body}
        return exc.code, payload


def run() -> int:
    args = _parse_args()
    settings = get_settings()
    seed_set = _load_seed_questions(args.questions)
    output_path = _resolve_repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger = create_run_logger(
        settings,
        "agent_ask_dataset_bootstrap",
        {
            "base_url": args.base_url,
            "questions": args.questions,
            "output": str(output_path),
            "top_k": args.top_k,
        },
    )

    cases: list[dict[str, Any]] = []
    endpoint = args.base_url.rstrip("/") + "/api/v1/agent/ask"

    for index, seed in enumerate(seed_set.questions, start=1):
        session_id = f"dataset-bootstrap-{index:02d}-{_slugify(seed.id)}"
        request_payload = {
            "session_id": session_id,
            "processed_file_path": seed_set.processed_file_path,
            "question": seed.question,
            "top_k": args.top_k,
        }
        logger.info("Bootstrapping case %s", seed.id)
        status_code, response_payload = _post_json(endpoint, request_payload)

        log_json_artifact(
            logger,
            f"cases/{seed.id}/request.json",
            request_payload,
        )
        log_json_artifact(
            logger,
            f"cases/{seed.id}/response.json",
            {"status_code": status_code, "payload": response_payload},
        )

        answer = str(response_payload.get("answer") or "")
        citations = list(response_payload.get("citations") or [])
        expected_pages = sorted(
            {
                int(citation["page_number"])
                for citation in citations
                if isinstance(citation, dict) and citation.get("page_number") is not None
            }
        )

        case = {
            "id": seed.id,
            "question": seed.question,
            "expected_route_strategy": response_payload.get("route_strategy"),
            "expected_pages": expected_pages,
            "expected_facts": _extract_expected_facts(answer),
            "notes": seed.notes
            or "Draft generated from live /agent/ask output. Review facts/pages before trusting as gold.",
            "draft_answer": answer,
            "draft_citations": citations,
            "draft_executed_steps": list(response_payload.get("executed_steps") or []),
            "bootstrap_status_code": status_code,
            "bootstrap_status": response_payload.get("status"),
        }
        cases.append(case)

    dataset = {
        "processed_file_path": seed_set.processed_file_path,
        "cases": cases,
    }
    output_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    log_json_artifact(logger, "draft_dataset.json", dataset)
    logger.info("Draft golden dataset written to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
