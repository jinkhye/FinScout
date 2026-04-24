from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from ...core.config import Settings


@dataclass(frozen=True)
class PipelineArtifacts:
    upload_path: Path
    pipeline_dir: Path
    processed_json: Path


def build_pipeline_artifacts(settings: Settings, pdf_name: str) -> PipelineArtifacts:
    pdf_slug = Path(pdf_name).with_suffix("").name or pdf_name
    output_dir = settings.pipeline_dir / pdf_slug
    return PipelineArtifacts(
        upload_path=settings.uploads_dir / pdf_name,
        pipeline_dir=output_dir,
        processed_json=output_dir / f"processed_{pdf_slug}.json",
    )


def write_json_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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
