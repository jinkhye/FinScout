from __future__ import annotations

from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse

from ...core.config import get_settings
from ...core.logger import create_run_logger
from ...dependencies import get_document_ingestion_service
from ...schemas.document import (
    DocumentProcessRequest,
    DocumentProcessResponse,
    DocumentReportResponse,
)
from ...services.document_processing.document_ingestion_service import (
    DocumentIngestionService,
)
from ...services.vector_ingestion.vector_index import load_processed_payload


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


@router.get("/report", response_model=DocumentReportResponse)
async def get_report_metadata(
    request: Request,
    processed_file_path: str = Query(..., min_length=1),
) -> DocumentReportResponse:
    settings = get_settings()
    try:
        processed_path, payload = load_processed_payload(settings, processed_file_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    pdf_name = str(payload.get("pdf_name") or "")
    if not pdf_name:
        raise HTTPException(
            status_code=400,
            detail="Processed file does not contain pdf_name",
        )

    pdf_path = settings.uploads_dir / pdf_name
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF not found for processed file: {pdf_path}",
        )

    pdf_base_url = str(request.url_for("get_report_pdf"))
    pdf_query = urlencode({"processed_file_path": str(processed_path)})
    return DocumentReportResponse(
        processed_file_path=str(processed_path),
        pdf_name=pdf_name,
        company_name=str(payload.get("company_name") or "unknown"),
        year=str(payload.get("year") or "unknown"),
        title=pdf_name.removesuffix(".pdf"),
        pdf_url=f"{pdf_base_url}?{pdf_query}",
    )


@router.get("/report/pdf")
async def get_report_pdf(
    processed_file_path: str = Query(..., min_length=1),
) -> FileResponse:
    settings = get_settings()
    try:
        _, payload = load_processed_payload(settings, processed_file_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    pdf_name = str(payload.get("pdf_name") or "")
    if not pdf_name:
        raise HTTPException(
            status_code=400,
            detail="Processed file does not contain pdf_name",
        )

    pdf_path = settings.uploads_dir / pdf_name
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF not found for processed file: {pdf_path}",
        )

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=pdf_name,
    )
