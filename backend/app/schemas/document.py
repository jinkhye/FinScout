from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


ProcessingStatus = Literal["success", "error"]


class TableSummary(BaseModel):
    table_index: int | None = None
    table_raw: str = ""
    table_summary: str = ""
    status: str = "unknown"
    error: str | None = None
    model: str | None = None


class PageOutput(BaseModel):
    page_number: int
    section: str = "unknown"
    has_table: bool = False
    tables: List[TableSummary] = Field(default_factory=list)
    markdown_raw: str = ""
    markdown_clean: str = ""


class DocumentProcessResponse(BaseModel):
    file_path: str = ""
    pdf: str = ""
    company_name: str = "unknown"
    year: str = "unknown"
    auditor_opinion: Literal["qualified", "unqualified", "unknown"] = "unknown"
    auditor_firm: str = "unknown"
    auditor_name: str = "unknown"
    audit_period: str = "unknown"
    total_pages_parsed: int = 0
    classified_pages: Dict[str, List[int]] = Field(default_factory=dict)
    pages: List[PageOutput] = Field(default_factory=list)
    status: ProcessingStatus = "error"
    error: str | None = None
    errors: List[str] = Field(default_factory=list)
