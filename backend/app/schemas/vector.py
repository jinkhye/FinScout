from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


VectorIngestStatus = Literal["success", "error"]


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
