from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from ..core.config import Settings
from .document_artifacts import normalize_classified_pages


PDF_NAME = "99SMART-Annual-Report-2024.pdf"
TABLE_SUMMARY_FALLBACK = "Table summary unavailable."

TABLE_SUMMARY_PROMPT = """You are a financial analyst assistant specializing in Malaysian corporate annual reports.

You are given a table extracted from the {section} section of a Malaysian corporate annual report.

Your task is to write a dense factual summary of this table that captures all key figures, metrics, comparisons, and trends across all periods shown.

Rules:
- Check the table header for the unit scale (e.g. RM'000, RM'million) and convert ALL figures to their true value throughout the summary (e.g. 100,000 in RM'000 = RM100,000,000)
- Do not interpret or give opinions, only describe what the table contains
- Write in present tense
- Write as many sentences as needed to capture all figures - do not truncate

Table:
{table_markdown}

Summary:"""


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


class AuditorResponse(BaseModel):
    company_name: str = Field(...)
    auditor_opinion: str = Field(...)
    auditor_firm: str = Field(...)
    auditor_name: str = Field(...)
    audit_period: str = Field(...)


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


HTML_TABLE_RE = re.compile(
    r"<\s*table\b[^>]*>[\s\S]*?<\s*/\s*table\s*>", re.IGNORECASE
)


def _is_markdown_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if len(cells) < 2:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def _is_markdown_table_row(line: str) -> bool:
    stripped = line.strip()
    return "|" in stripped and not stripped.startswith("<!--")


def _markdown_table_spans(markdown: str) -> List[Tuple[int, int]]:
    lines = markdown.splitlines(keepends=True)
    spans: List[Tuple[int, int]] = []
    offset = 0
    index = 0

    while index < len(lines) - 1:
        header = lines[index]
        separator = lines[index + 1]
        if not (
            _is_markdown_table_row(header)
            and _is_markdown_table_separator(separator)
        ):
            offset += len(header)
            index += 1
            continue

        start = offset
        end = offset + len(header) + len(separator)
        index += 2
        offset = end

        while index < len(lines) and _is_markdown_table_row(lines[index]):
            end += len(lines[index])
            offset += len(lines[index])
            index += 1

        spans.append((start, end))

    return spans


def _table_block_spans(markdown: str) -> List[Tuple[int, int]]:
    spans = [match.span() for match in HTML_TABLE_RE.finditer(markdown)]

    for markdown_start, markdown_end in _markdown_table_spans(markdown):
        if any(
            markdown_start >= html_start and markdown_end <= html_end
            for html_start, html_end in spans
        ):
            continue
        spans.append((markdown_start, markdown_end))

    return sorted(spans)


def extract_table_blocks(markdown: str) -> List[str]:
    return [markdown[start:end].strip() for start, end in _table_block_spans(markdown)]


def replace_table_blocks_with_placeholders(markdown: str) -> str:
    spans = _table_block_spans(markdown)
    if not spans:
        return markdown

    parts: List[str] = []
    cursor = 0
    for index, (start, end) in enumerate(spans, start=1):
        parts.append(markdown[cursor:start])
        parts.append(f" TABLESUMMARYSLOT{index} ")
        cursor = end

    parts.append(markdown[cursor:])
    return "".join(parts)


def inject_table_summaries(cleaned_text: str, summaries: List[str]) -> str:
    out = cleaned_text
    for index, summary in enumerate(summaries, start=1):
        out = out.replace(f"TABLESUMMARYSLOT{index}", summary)
    return re.sub(r"\s+", " ", out).strip()


def get_gemini_client() -> Optional[Any]:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def summarize_table_with_gemini(
    client: Optional[Any],
    settings: Settings,
    section: str,
    page_number: int,
    table_index: int,
    table_markdown: str,
) -> Tuple[str, str, Optional[str]]:
    if client is None:
        return TABLE_SUMMARY_FALLBACK, "fallback", "Gemini client unavailable"

    prompt = TABLE_SUMMARY_PROMPT.format(
        section=section,
        table_markdown=table_markdown,
    )

    last_error: Optional[str] = None
    total_attempts = settings.gemini_max_retries + 1
    for attempt in range(1, total_attempts + 1):
        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config={"thinking_config": {"thinking_level": "low"}},
            )
            text = (getattr(response, "text", None) or "").strip()
            if not text:
                last_error = "Empty Gemini response"
                raise ValueError(last_error)
            return text, "ok" if attempt == 1 else "ok_retry", None
        except Exception as exc:
            last_error = str(exc)
            if attempt < total_attempts:
                time.sleep(settings.gemini_retry_delay_sec * attempt)
                continue

    return TABLE_SUMMARY_FALLBACK, "error", last_error


def load_auditor_pages(source_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    pages = source_payload.get("pages", [])
    if not isinstance(pages, list):
        raise ValueError("Source payload pages field must be a list")

    auditor_pages: List[Dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        if page.get("section") != "auditor_report":
            continue
        page_number = page.get("page_number")
        if isinstance(page_number, bool) or not isinstance(page_number, int):
            continue
        auditor_pages.append(page)

    auditor_pages.sort(key=lambda page: int(page["page_number"]))
    if not auditor_pages:
        raise ValueError("No auditor_report pages found in the source payload")
    return auditor_pages


def build_consolidated_auditor_markdown(pages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for page in pages:
        page_number = int(page["page_number"])
        section = page.get("section", "auditor_report")
        parts.append(f"<!-- Page {page_number} | section={section} -->")
        parts.append("")
        parts.append(page.get("markdown_clean", ""))
        parts.append("")
        parts.append("---")
        parts.append("")

    return "\n".join(parts).strip() + "\n"


AUDITOR_PROMPT = dedent(
    """
    Analyze the auditor's report and extract:
    - company_name: Full legal name of the company being audited, or 'unknown' if not specified
    - auditor_opinion: Auditor opinion type (only 'qualified', 'unqualified', or 'unknown')
    - auditor_firm: Name of the auditing firm (e.g., Deloitte, PwC), or 'unknown' if not specified
    - auditor_name: Name of the individual auditor signing the report, or 'unknown' if not specified
    - audit_period: Period covered by the audit in DD-MM-YYYY format (e.g., '31-12-2024'), or 'unknown' if not specified

    Auditor report:
    {auditor_report_markdown}
    """
).strip()


def call_auditor_extraction_model(
    client: Optional[Any],
    model: str,
    consolidated_markdown: str,
) -> AuditorResponse:
    if client is None:
        raise RuntimeError(
            "Gemini client unavailable: set GEMINI_API_KEY or GOOGLE_API_KEY"
        )

    prompt = AUDITOR_PROMPT.format(auditor_report_markdown=consolidated_markdown)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=AuditorResponse.model_json_schema(),
        ),
    )

    response_text = (getattr(response, "text", None) or "").strip()
    if not response_text:
        raise ValueError("Gemini returned an empty auditor extraction response")

    return AuditorResponse.model_validate_json(response_text)


def derive_year(audit_period: str) -> str:
    if audit_period == "unknown":
        return "unknown"

    for format_string in ("%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(audit_period, format_string).strftime("%Y")
        except ValueError:
            continue
    return "unknown"


def build_auditor_output_payload(
    source_payload: Dict[str, Any],
    auditor_response: AuditorResponse,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "pdf": source_payload.get("pdf", PDF_NAME),
        "company_name": auditor_response.company_name,
        "year": derive_year(auditor_response.audit_period),
        "auditor_opinion": auditor_response.auditor_opinion,
        "auditor_firm": auditor_response.auditor_firm,
        "auditor_name": auditor_response.auditor_name,
        "audit_period": auditor_response.audit_period,
    }

    for key, value in source_payload.items():
        if key == "pdf":
            continue
        payload[key] = value

    return payload


def write_debug_input(consolidated_markdown: str, debug_input_md: Path) -> None:
    debug_input_md.parent.mkdir(parents=True, exist_ok=True)
    debug_input_md.write_text(consolidated_markdown, encoding="utf-8")


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
