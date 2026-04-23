from __future__ import annotations

from fastapi import APIRouter, Depends

from ...core.config import get_settings
from ...core.logger import create_run_logger
from ...dependencies import get_query_planner_service
from ...schemas.query import QueryPlanRequest, QueryPlanResponse
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
