from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def load_json_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON payload not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


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


def available_case_stems(cases_dir: Path) -> List[str]:
    stems = set()
    for suffix in ("cleaned_pages", "parsed_pages", "raw_pages", "auditor_report"):
        suffix_glob = f"*.{suffix}.json"
        for path in cases_dir.glob(suffix_glob):
            name = path.name[: -len(f".{suffix}.json")]
            stems.add(name)
    return sorted(stems)


def resolve_case_stem(
    cases_dir: Path, uploaded_filename: str, default_case_stem: str
) -> str:
    upload_stem = Path(uploaded_filename).stem
    candidate_stems = set(available_case_stems(cases_dir))

    if upload_stem in candidate_stems:
        return upload_stem

    if default_case_stem in candidate_stems:
        return default_case_stem

    if len(candidate_stems) == 1:
        return next(iter(candidate_stems))

    if candidate_stems:
        joined = ", ".join(sorted(candidate_stems))
        raise FileNotFoundError(
            f"No case artifacts found for {upload_stem!r}. Available stems: {joined}"
        )

    raise FileNotFoundError("No case artifacts are available in the cases directory")


def load_document_payload(
    cases_dir: Path, case_stem: str
) -> Tuple[Dict[str, Any], Path]:
    for suffix in ("cleaned_pages", "parsed_pages", "raw_pages"):
        candidate = cases_dir / f"{case_stem}.{suffix}.json"
        if candidate.exists():
            return load_json_payload(candidate), candidate

    raise FileNotFoundError(
        f"No cleaned, parsed, or raw page payload found for case stem {case_stem!r}"
    )


def load_auditor_payload(cases_dir: Path, case_stem: str) -> Dict[str, Any] | None:
    candidate = cases_dir / f"{case_stem}.auditor_report.json"
    if not candidate.exists():
        return None
    return load_json_payload(candidate)


def page_numbers_from_classified_pages(
    classified_pages: Dict[str, List[int]],
) -> List[int]:
    page_numbers: List[int] = []
    for pages in classified_pages.values():
        page_numbers.extend(pages)
    return sorted(set(page_numbers))


def select_pages(
    pages: Sequence[Dict[str, Any]],
    classified_pages: Dict[str, List[int]],
) -> Tuple[List[Dict[str, Any]], List[int]]:
    allowed_pages = set(page_numbers_from_classified_pages(classified_pages))
    selected: List[Dict[str, Any]] = []
    found_pages: set[int] = set()

    for page in pages:
        page_number = page.get("page_number")
        if isinstance(page_number, bool) or not isinstance(page_number, int):
            continue
        if page_number in allowed_pages:
            selected.append(page)
            found_pages.add(page_number)

    missing_pages = sorted(allowed_pages - found_pages)
    selected.sort(key=lambda page: int(page["page_number"]))
    return selected, missing_pages
