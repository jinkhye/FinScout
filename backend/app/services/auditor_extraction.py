from __future__ import annotations

from datetime import datetime
from textwrap import dedent
from typing import Any, Dict, List, Optional

from google.genai import types
from pydantic import BaseModel, Field


PDF_NAME = "99SMART-Annual-Report-2024.pdf"


class AuditorResponse(BaseModel):
    company_name: str = Field(...)
    auditor_opinion: str = Field(...)
    auditor_firm: str = Field(...)
    auditor_name: str = Field(...)
    audit_period: str = Field(...)


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
