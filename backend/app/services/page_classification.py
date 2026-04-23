from __future__ import annotations

from typing import Any, Dict, List


def normalize_classified_pages(classified_pages: Any) -> Dict[str, List[int]]:
    if not isinstance(classified_pages, dict):
        raise ValueError("classified_pages must be a JSON object")

    normalized: Dict[str, List[int]] = {}
    for section, pages in classified_pages.items():
        if not isinstance(section, str):
            raise ValueError("classified_pages keys must be strings")
        if not isinstance(pages, list):
            raise ValueError(f"{section}: pages must be a list")

        cleaned_pages: List[int] = []
        for page_number in pages:
            if isinstance(page_number, bool) or not isinstance(page_number, int):
                raise ValueError(
                    f"{section}: page numbers must be integers, got {page_number!r}"
                )
            if page_number < 1:
                raise ValueError(
                    f"{section}: page numbers must be >= 1, got {page_number}"
                )
            cleaned_pages.append(page_number)

        normalized[section] = sorted(set(cleaned_pages))

    return normalized

