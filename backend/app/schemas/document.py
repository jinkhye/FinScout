from __future__ import annotations

from typing import Annotated, Dict, List, Literal

from pydantic import BaseModel, Field


ProcessingStatus = Literal["success", "error"]
PageNumber = Annotated[int, Field(gt=0, strict=True)]


CLASSIFIED_PAGES_EXAMPLE: Dict[str, List[int]] = {
    "company_overview": [
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
    ],
    "mda": [24, 25, 31, 32, 33, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
    "auditor_report": [143, 144, 145, 146, 147],
    "balance_sheet": [148],
    "income_statement": [149],
    "equitychange_statement": [150, 151],
    "cashflow_statement": [152, 153],
    "notes": [
        154,
        155,
        156,
        157,
        158,
        159,
        160,
        161,
        162,
        163,
        164,
        165,
        166,
        167,
        168,
        169,
        170,
        171,
        172,
        173,
        174,
        175,
        176,
        177,
        178,
        179,
        180,
        181,
        182,
        183,
        184,
        185,
        186,
        187,
        188,
    ],
}


class DocumentProcessRequest(BaseModel):
    file_path: str = Field(
        ...,
        examples=["uploads/99SMART-Annual-Report-2024.pdf"],
    )
    classified_pages: Dict[str, List[PageNumber]] = Field(
        ...,
        examples=[CLASSIFIED_PAGES_EXAMPLE],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "file_path": "uploads/99SMART-Annual-Report-2024.pdf",
                    "classified_pages": CLASSIFIED_PAGES_EXAMPLE,
                }
            ]
        }
    }


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
    pdf_name: str = ""
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
