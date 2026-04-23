from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from google import genai

from ..core.config import Settings
from .document_artifacts import normalize_classified_pages


@dataclass(frozen=True)
class PipelineArtifacts:
    upload_path: Path
    pipeline_dir: Path
    raw_json: Path
    parsed_json: Path
    cleaned_json: Path
    auditor_json: Path
    debug_dir: Path
    debug_input_md: Path


def build_pipeline_artifacts(settings: Settings, pdf_name: str) -> PipelineArtifacts:
    pdf_stem = Path(pdf_name).stem or pdf_name
    pipeline_dir = settings.pipeline_dir / pdf_stem
    debug_dir = pipeline_dir / "debug"
    return PipelineArtifacts(
        upload_path=settings.uploads_dir / pdf_name,
        pipeline_dir=pipeline_dir,
        raw_json=pipeline_dir / "raw_pages.json",
        parsed_json=pipeline_dir / "parsed_pages.json",
        cleaned_json=pipeline_dir / "cleaned_pages.json",
        auditor_json=pipeline_dir / "auditor_report.json",
        debug_dir=debug_dir,
        debug_input_md=debug_dir / "auditor_report.input.md",
    )


def write_json_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_json_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON payload not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def load_existing_pages(path: Path, expected_pdf: str) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}

    existing = load_json_payload(path)
    existing_pdf = existing.get("pdf")
    if existing_pdf not in (None, expected_pdf):
        raise ValueError(
            f"Existing output belongs to a different PDF: {existing_pdf!r}"
        )

    pages_raw = existing.get("pages", [])
    if not isinstance(pages_raw, list):
        raise ValueError("Existing output pages field must be a list")

    pages_by_number: Dict[int, Dict[str, Any]] = {}
    for page in pages_raw:
        if not isinstance(page, dict):
            continue
        page_number = page.get("page_number")
        if isinstance(page_number, bool) or not isinstance(page_number, int):
            continue
        pages_by_number[page_number] = page

    return pages_by_number


def load_raw_pages(path: Path) -> Tuple[Dict[str, List[int]], List[Dict[str, Any]]]:
    payload = load_json_payload(path)

    classified = payload.get("classified_pages")
    pages = payload.get("pages")
    if not isinstance(classified, dict):
        raise ValueError("Raw pages file is missing classified_pages object")
    if not isinstance(pages, list):
        raise ValueError("Raw pages file is missing pages list")

    raw_pages: List[Dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        page_number = page.get("page_number")
        if isinstance(page_number, bool) or not isinstance(page_number, int):
            continue
        raw_pages.append(page)

    raw_pages.sort(key=lambda page: int(page["page_number"]))
    normalized_classified = normalize_classified_pages(classified)
    return normalized_classified, raw_pages


def build_page_to_section_map(
    classified: Dict[str, List[int]],
) -> Tuple[Dict[int, str], List[Tuple[int, List[str]]]]:
    page_to_section: Dict[int, str] = {}
    duplicates: Dict[int, List[str]] = {}

    for section, pages in classified.items():
        for page in pages:
            if page in page_to_section:
                duplicates.setdefault(page, [page_to_section[page]]).append(section)
            else:
                page_to_section[page] = section

    return page_to_section, sorted(
        (page, sections) for page, sections in duplicates.items()
    )


def page_numbers_from_classified_pages(
    classified_pages: Dict[str, List[int]],
) -> List[int]:
    page_numbers: List[int] = []
    for pages in classified_pages.values():
        page_numbers.extend(pages)
    return sorted(set(page_numbers))


def pages_to_target_pages(pages: List[int]) -> str:
    if not pages:
        return ""

    ranges: List[str] = []
    start = prev = pages[0]

    for page in pages[1:]:
        if page == prev + 1:
            prev = page
            continue

        if start == prev:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{prev}")
        start = prev = page

    if start == prev:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{prev}")

    return ",".join(ranges)


def chunk_pages(pages: Sequence[int], batch_size: int) -> List[List[int]]:
    if batch_size <= 0:
        return [list(pages)]
    return [list(pages[i : i + batch_size]) for i in range(0, len(pages), batch_size)]


def clean_markdown_for_embedding(markdown: str) -> str:
    text = markdown
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = text.replace("`", "")
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*_]{3,}\s*$", " ", text)
    text = re.sub(r"(\*\*|__|\*|_|~~)", "", text)
    text = re.sub(r"(?m)^\s*>\s?", "", text)
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)
    text = re.sub(r"(?m)^\s*\d+\.\s+", "", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_gemini_client() -> Optional[Any]:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def write_debug_markdown_chunks(
    debug_dir: Path,
    pages: List[Dict[str, Any]],
    chunk_size: int,
    raw_key: str,
    clean_key: str,
) -> None:
    if not pages:
        return

    debug_dir.mkdir(parents=True, exist_ok=True)

    for offset in range(0, len(pages), chunk_size):
        chunk = pages[offset : offset + chunk_size]
        start = int(chunk[0]["page_number"])
        end = int(chunk[-1]["page_number"])
        file_path = debug_dir / f"page_{start}-{end}.debug.md"

        parts: List[str] = []
        for page in chunk:
            page_number = page.get("page_number")
            section = page.get("section", "unknown")
            parts.append(f"<!-- Page {page_number} | section={section} -->")
            parts.append("")
            parts.append("## RAW")
            parts.append("")
            parts.append(page.get(raw_key, ""))
            parts.append("")
            parts.append("## CLEAN")
            parts.append("")
            parts.append(page.get(clean_key, ""))
            parts.append("")
            parts.append("---")
            parts.append("")

        file_path.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")
