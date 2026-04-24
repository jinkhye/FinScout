from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator

from .query import QueryPlanResponse, RouteStrategy


AgentAskStatus = Literal["success", "error"]


class AgentAskRequest(BaseModel):
    processed_file_path: str = Field(
        ...,
        examples=[
            "backend/storage/pipelines/99SMART-Annual-Report-2024/processed_99SMART-Annual-Report-2024.json"
        ],
    )
    collection_name: str | None = Field(
        default=None,
        examples=["report_99smart_annual_report_2024"],
    )
    question: str = Field(
        ..., min_length=1, examples=["What was the total revenue in 2024?"]
    )
    top_k: int = Field(default=8, ge=1, le=20)
    rerank: bool = False

    @field_validator("processed_file_path", "collection_name", "question")
    @classmethod
    def must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field cannot be blank")
        return cleaned

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "processed_file_path": "backend/storage/pipelines/99SMART-Annual-Report-2024/processed_99SMART-Annual-Report-2024.json",
                    "question": "What was the total revenue in 2024?",
                    "top_k": 8,
                    "rerank": False,
                }
            ]
        }
    }


class AgentCitation(BaseModel):
    page_number: int
    section: str
    excerpt: str = ""


class AgentAskResponse(BaseModel):
    question: str = ""
    answer: str = ""
    company_name: str = "unknown"
    year: str = "unknown"
    route_strategy: RouteStrategy = "vector_search"
    reranked: bool = False
    citations: List[AgentCitation] = Field(default_factory=list)
    planner: QueryPlanResponse | None = None
    status: AgentAskStatus = "error"
    error: str | None = None
    errors: List[str] = Field(default_factory=list)
