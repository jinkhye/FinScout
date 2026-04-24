from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator

from .vector import SectionLabel


QueryPlanStatus = Literal["success", "error"]
QueryContextStatus = Literal["success", "error"]
RouteStrategy = Literal["vector_search", "full_context"]


class QueryPlanRequest(BaseModel):
    processed_file_path: str = Field(
        ...,
        examples=[
            "backend/storage/pipelines/99SMART-Annual-Report-2024/processed_99SMART-Annual-Report-2024.json"
        ],
    )
    query: str = Field(..., min_length=1, examples=["wat was revnue?"])
    conversation_context: str = ""

    @field_validator("processed_file_path", "query")
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
                    "processed_file_path": "backend/storage/pipelines/99SMART-Annual-Report-2024/processed_99SMART-Annual-Report-2024.json",
                    "query": "wat was revnue?",
                }
            ]
        }
    }


class QueryPlannerModelOutput(BaseModel):
    optimized_query: str = Field(...)
    selected_sections: List[SectionLabel] = Field(default_factory=list)


class QueryPlanResponse(BaseModel):
    original_query: str = ""
    optimized_query: str = ""
    company_name: str = "unknown"
    year: str = "unknown"
    selected_sections: List[SectionLabel] = Field(default_factory=list)
    route_strategy: RouteStrategy = "vector_search"
    vector_search_sections: List[SectionLabel] = Field(default_factory=list)
    full_context_sections: List[SectionLabel] = Field(default_factory=list)
    status: QueryPlanStatus = "error"
    error: str | None = None
    errors: List[str] = Field(default_factory=list)


class QueryContextRequest(BaseModel):
    processed_file_path: str = Field(
        ...,
        examples=[
            "backend/storage/pipelines/99SMART-Annual-Report-2024/processed_99SMART-Annual-Report-2024.json"
        ],
    )
    sections: List[SectionLabel] = Field(
        ...,
        examples=[["auditor_report"]],
    )

    @field_validator("processed_file_path")
    @classmethod
    def path_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field cannot be blank")
        return cleaned

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "processed_file_path": "backend/storage/pipelines/99SMART-Annual-Report-2024/processed_99SMART-Annual-Report-2024.json",
                    "sections": ["auditor_report"],
                }
            ]
        }
    }


class QueryContextResponse(BaseModel):
    processed_file_path: str = ""
    company_name: str = "unknown"
    year: str = "unknown"
    sections: List[SectionLabel] = Field(default_factory=list)
    pages_count: int = 0
    context_text: str = ""
    status: QueryContextStatus = "error"
    error: str | None = None
    errors: List[str] = Field(default_factory=list)
