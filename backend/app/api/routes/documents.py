from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ...dependencies import get_document_ingestion_service
from ...schemas.document import DocumentProcessResponse
from ...services.document_ingestion_service import DocumentIngestionService


router = APIRouter()


@router.post("/process", response_model=DocumentProcessResponse)
async def process_document(
    file: UploadFile = File(...),
    classified_pages_json: str = Form(...),
    service: DocumentIngestionService = Depends(get_document_ingestion_service),
) -> DocumentProcessResponse:
    try:
        classified_pages_raw = json.loads(classified_pages_json)
    except json.JSONDecodeError as exc:
        return DocumentProcessResponse(
            pdf=file.filename or "",
            status="error",
            error=f"Invalid classified_pages_json: {exc.msg}",
            errors=[f"Invalid classified_pages_json: {exc.msg}"],
        )

    return await service.process_upload(file, classified_pages_raw)
