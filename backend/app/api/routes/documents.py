from __future__ import annotations

from fastapi import APIRouter, Depends

from ...dependencies import get_document_ingestion_service
from ...schemas.document import DocumentProcessRequest, DocumentProcessResponse
from ...services.document_ingestion_service import DocumentIngestionService


router = APIRouter()


@router.post("/process", response_model=DocumentProcessResponse)
async def process_document(
    request: DocumentProcessRequest,
    service: DocumentIngestionService = Depends(get_document_ingestion_service),
) -> DocumentProcessResponse:
    return await service.process_file(request.file_path, request.classified_pages)
