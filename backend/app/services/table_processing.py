from __future__ import annotations

import re
import time
from typing import Any, List, Optional, Tuple

from ..core.config import Settings


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
