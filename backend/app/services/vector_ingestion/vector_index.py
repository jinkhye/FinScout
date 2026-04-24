from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ...core.config import Settings


class VectorIndexRecord(BaseModel):
    processed_file_path: str
    pdf_name: str
    company_name: str
    year: str
    collection_name: str
    embedding_model: str
    qdrant_url: str
    qdrant_distance: str
    points_inserted: int
    skipped_pages: list[int]
    status: str
    updated_at: str


def resolve_processed_file_path(settings: Settings, processed_file_path: str) -> Path:
    candidate = Path(processed_file_path)
    if not candidate.is_absolute():
        candidate = settings.repository_root / candidate
    candidate = candidate.resolve()

    if not candidate.exists():
        raise FileNotFoundError(f"Processed file not found: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"Processed path must point to a file: {candidate}")
    return candidate


def load_processed_payload(settings: Settings, processed_file_path: str) -> tuple[Path, dict[str, Any]]:
    path = resolve_processed_file_path(settings, processed_file_path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("Processed file must contain a JSON object")
    if not isinstance(payload.get("pages"), list):
        raise ValueError("Processed file must contain a pages list")
    return path, payload


def vector_index_path(processed_path: Path) -> Path:
    return processed_path.parent / "vector_index.json"


def vector_runs_path(processed_path: Path) -> Path:
    return processed_path.parent / "vector_runs.json"


def write_vector_index(processed_path: Path, record: VectorIndexRecord) -> None:
    path = vector_index_path(processed_path)
    path.write_text(
        record.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def append_vector_run(processed_path: Path, record: VectorIndexRecord) -> None:
    path = vector_runs_path(processed_path)
    runs: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
        if isinstance(loaded, list):
            runs = [item for item in loaded if isinstance(item, dict)]

    runs.append(record.model_dump())
    path.write_text(
        json.dumps(runs, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_vector_index_record(
    settings: Settings,
    processed_file_path: str,
) -> tuple[Path, VectorIndexRecord]:
    processed_path = resolve_processed_file_path(settings, processed_file_path)
    path = vector_index_path(processed_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Vector index not found for processed file: {path}. Run /api/v1/vector/ingest first."
        )

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"Vector index must contain a JSON object: {path}")
    return processed_path, VectorIndexRecord.model_validate(payload)


def resolve_collection_name(
    settings: Settings,
    processed_file_path: str | None,
    collection_name: str | None,
) -> str:
    if collection_name and collection_name.strip():
        return collection_name.strip()
    if not processed_file_path:
        raise ValueError("processed_file_path is required when collection_name is not provided")
    _, record = load_vector_index_record(settings, processed_file_path)
    return record.collection_name


def build_vector_index_record(
    settings: Settings,
    processed_path: Path,
    processed_payload: dict[str, Any],
    collection_name: str,
    points_inserted: int,
    skipped_pages: list[int],
) -> VectorIndexRecord:
    return VectorIndexRecord(
        processed_file_path=str(processed_path),
        pdf_name=str(processed_payload.get("pdf_name") or ""),
        company_name=str(processed_payload.get("company_name") or "unknown"),
        year=str(processed_payload.get("year") or "unknown"),
        collection_name=collection_name,
        embedding_model=settings.gemini_embedding_model,
        qdrant_url=settings.qdrant_url,
        qdrant_distance=settings.qdrant_distance,
        points_inserted=points_inserted,
        skipped_pages=skipped_pages,
        status="success",
        updated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
