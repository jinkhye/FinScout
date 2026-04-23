from __future__ import annotations

from typing import Any, List, Literal

from pydantic import BaseModel, Field, field_validator


VectorIngestStatus = Literal["success", "error"]
VectorQueryStatus = Literal["success", "error"]
SectionLabel = Literal[
    "company_overview",
    "mda",
    "auditor_report",
    "balance_sheet",
    "income_statement",
    "equitychange_statement",
    "cashflow_statement",
    "notes",
    "other",
]


class VectorIngestRequest(BaseModel):
    processed_file_path: str = Field(
        ...,
        examples=[
            "backend/storage/pipelines/99SMART-Annual-Report-2024/processed_99SMART-Annual-Report-2024.json"
        ],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "processed_file_path": "backend/storage/pipelines/99SMART-Annual-Report-2024/processed_99SMART-Annual-Report-2024.json"
                }
            ]
        }
    }


class VectorIngestResponse(BaseModel):
    collection_name: str = ""
    points_inserted: int = 0
    status: VectorIngestStatus = "error"
    error: str | None = None
    errors: List[str] = Field(default_factory=list)


class VectorQueryFilters(BaseModel):
    sections: List[SectionLabel] = Field(default_factory=list)
    company_name: str | None = None
    year: str | None = None
    has_table: bool | None = None


class VectorQueryRequest(BaseModel):
    collection_name: str = Field(
        ...,
        min_length=1,
        examples=["report_99smart_annual_report_2024"],
    )
    query: str = Field(
        ...,
        min_length=1,
        examples=["What was the total revenue in 2024?"],
    )
    top_k: int = Field(default=5, ge=1, le=20)
    filters: VectorQueryFilters | None = None

    @field_validator("collection_name", "query")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field cannot be blank")
        return cleaned

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "collection_name": "report_99smart_annual_report_2024",
                    "query": "What was the total revenue in 2024?",
                    "top_k": 5,
                    "filters": {
                        "sections": ["income_statement"],
                        "company_name": "99 Speed Mart Retail Holdings Berhad",
                        "year": "2024",
                        "has_table": True,
                    },
                }
            ]
        }
    }


class VectorQueryResult(BaseModel):
    score: float
    pdf_name: str = ""
    company_name: str = ""
    year: str = ""
    page_number: int | None = None
    section: str = ""
    has_table: bool = False
    text: str = ""
    text_for_embedding: str = ""
    tables: List[Any] = Field(default_factory=list)


class VectorQueryResponse(BaseModel):
    collection_name: str = ""
    query: str = ""
    top_k: int = 5
    results_count: int = 0
    results: List[VectorQueryResult] = Field(default_factory=list)
    status: VectorQueryStatus = "error"
    error: str | None = None
    errors: List[str] = Field(default_factory=list)
