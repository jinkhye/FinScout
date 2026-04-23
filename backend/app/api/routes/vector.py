from __future__ import annotations

from fastapi import APIRouter, Depends

from ...core.config import get_settings
from ...core.logger import create_run_logger
from ...dependencies import get_vector_ingestion_service
from ...schemas.vector import VectorIngestRequest, VectorIngestResponse
from ...services.vector_ingestion.vector_ingestion_service import VectorIngestionService


router = APIRouter()


@router.post("/ingest", response_model=VectorIngestResponse)
async def ingest_vectors(
    request: VectorIngestRequest,
    service: VectorIngestionService = Depends(get_vector_ingestion_service),
) -> VectorIngestResponse:
    logger = create_run_logger(
        get_settings(),
        "vector_ingest",
        request.model_dump(),
    )
    return await service.ingest_processed_file(
        request.processed_file_path,
        logger=logger,
    )
