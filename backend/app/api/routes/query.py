from __future__ import annotations

from fastapi import APIRouter, Depends

from ...core.config import get_settings
from ...core.logger import create_run_logger
from ...dependencies import get_query_context_service, get_query_planner_service
from ...schemas.query import (
    QueryContextRequest,
    QueryContextResponse,
    QueryPlanRequest,
    QueryPlanResponse,
)
from ...services.query_planning.query_context_service import QueryContextService
from ...services.query_planning.query_planner_service import QueryPlannerService


router = APIRouter()


@router.post("/plan", response_model=QueryPlanResponse)
async def plan_query(
    request: QueryPlanRequest,
    service: QueryPlannerService = Depends(get_query_planner_service),
) -> QueryPlanResponse:
    logger = create_run_logger(
        get_settings(),
        "query_plan",
        request.model_dump(),
    )
    return await service.plan(request, logger=logger)


@router.post("/context", response_model=QueryContextResponse)
async def load_query_context(
    request: QueryContextRequest,
    service: QueryContextService = Depends(get_query_context_service),
) -> QueryContextResponse:
    logger = create_run_logger(
        get_settings(),
        "query_context",
        request.model_dump(),
    )
    return await service.load_context(request, logger=logger)
