from __future__ import annotations

from fastapi import APIRouter, Depends

from ...core.config import get_settings
from ...core.logger import create_run_logger
from ...dependencies import get_document_ingestion_service
from ...schemas.document import DocumentProcessRequest, DocumentProcessResponse
from ...services.document_processing.document_ingestion_service import (
    DocumentIngestionService,
)


router = APIRouter()


@router.post("/process", response_model=DocumentProcessResponse)
async def process_document(
    request: DocumentProcessRequest,
    service: DocumentIngestionService = Depends(get_document_ingestion_service),
) -> DocumentProcessResponse:
    logger = create_run_logger(
        get_settings(),
        "documents_process",
        request.model_dump(),
    )
    return await service.process_file(
        request.file_path,
        request.classified_pages,
        logger=logger,
    )
