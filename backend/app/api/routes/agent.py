from __future__ import annotations

from fastapi import APIRouter, Depends

from ...core.config import get_settings
from ...core.logger import create_run_logger
from ...dependencies import get_agent_service
from ...schemas.agent import AgentAskRequest, AgentAskResponse
from ...services.agent.agent_service import AgentService


router = APIRouter()


@router.post("/ask", response_model=AgentAskResponse)
async def ask_agent(
    request: AgentAskRequest,
    service: AgentService = Depends(get_agent_service),
) -> AgentAskResponse:
    logger = create_run_logger(
        get_settings(),
        "agent_ask",
        request.model_dump(),
    )
    return await service.ask(request, logger=logger)
