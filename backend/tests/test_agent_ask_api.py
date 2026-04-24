from __future__ import annotations

from backend.app.schemas.agent import AgentAskResponse, AgentCitation, AgentExecutedStep
from backend.app.schemas.query import QueryPlanResponse


VALID_REQUEST = {
    "session_id": "demo-session-001",
    "processed_file_path": "backend/storage/pipelines/99SMART-Annual-Report-2024/processed_99SMART-Annual-Report-2024.json",
    "question": "What was the total revenue in 2024?",
    "top_k": 8,
}


def _planner(route_strategy: str | None) -> QueryPlanResponse:
    return QueryPlanResponse(
        original_query=VALID_REQUEST["question"],
        intent="report_question" if route_strategy is not None else "direct_reply",
        optimized_query=VALID_REQUEST["question"],
        company_name="99 Speed Mart Retail Holdings Berhad",
        year="2024",
        route_strategy=route_strategy,
        status="success",
        errors=[],
    )


def test_agent_ask_rejects_blank_session_id(client):
    response = client.post(
        "/api/v1/agent/ask",
        json={**VALID_REQUEST, "session_id": "   "},
    )
    assert response.status_code == 422


def test_agent_ask_rejects_blank_processed_file_path(client):
    response = client.post(
        "/api/v1/agent/ask",
        json={**VALID_REQUEST, "processed_file_path": "   "},
    )
    assert response.status_code == 422


def test_agent_ask_rejects_blank_question(client):
    response = client.post(
        "/api/v1/agent/ask",
        json={**VALID_REQUEST, "question": "   "},
    )
    assert response.status_code == 422


def test_agent_ask_rejects_invalid_top_k(client):
    response = client.post(
        "/api/v1/agent/ask",
        json={**VALID_REQUEST, "top_k": 0},
    )
    assert response.status_code == 422


def test_agent_ask_direct_reply_success(client, override_agent_service):
    def handler(**_kwargs):
        return AgentAskResponse(
            session_id=VALID_REQUEST["session_id"],
            turn_index=1,
            question="hello",
            answer="Hello! Ask me anything about the annual report.",
            company_name="99 Speed Mart Retail Holdings Berhad",
            year="2024",
            route_strategy=None,
            reranked=False,
            retried=False,
            final_query="hello",
            executed_steps=[],
            citations=[],
            planner=_planner(None),
            status="success",
            errors=[],
        )

    override_agent_service(handler)
    response = client.post(
        "/api/v1/agent/ask",
        json={**VALID_REQUEST, "question": "hello"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["citations"] == []
    assert payload["executed_steps"] == []
    assert payload["route_strategy"] is None


def test_agent_ask_full_context_success(client, override_agent_service):
    def handler(**_kwargs):
        return AgentAskResponse(
            session_id=VALID_REQUEST["session_id"],
            turn_index=1,
            question=VALID_REQUEST["question"],
            answer="The total revenue in 2024 was RM 9,981,642,000.",
            company_name="99 Speed Mart Retail Holdings Berhad",
            year="2024",
            route_strategy="full_context",
            reranked=False,
            retried=False,
            final_query="total revenue 2024",
            executed_steps=[
                AgentExecutedStep(
                    step_index=1,
                    goal="Find total revenue",
                    query="total revenue 2024",
                    route_strategy="full_context",
                    selected_sections=["income_statement"],
                    cited_pages=[149],
                )
            ],
            citations=[
                AgentCitation(
                    page_number=149,
                    section="income_statement",
                    excerpt="Revenue 9,981,642",
                )
            ],
            planner=_planner("full_context"),
            status="success",
            errors=[],
        )

    override_agent_service(handler)
    response = client.post("/api/v1/agent/ask", json=VALID_REQUEST)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["route_strategy"] == "full_context"
    assert payload["citations"]
    assert payload["executed_steps"][0]["route_strategy"] == "full_context"


def test_agent_ask_vector_search_success(client, override_agent_service):
    def handler(**_kwargs):
        return AgentAskResponse(
            session_id=VALID_REQUEST["session_id"],
            turn_index=1,
            question="How many stores did the group operate?",
            answer="The group operated 2,833 stores in Malaysia.",
            company_name="99 Speed Mart Retail Holdings Berhad",
            year="2024",
            route_strategy="vector_search",
            reranked=True,
            retried=False,
            final_query="store count malaysia",
            executed_steps=[
                AgentExecutedStep(
                    step_index=1,
                    goal="Find store count",
                    query="store count malaysia",
                    route_strategy="vector_search",
                    selected_sections=["company_overview"],
                    reranked=True,
                    cited_pages=[17],
                )
            ],
            citations=[
                AgentCitation(
                    page_number=17,
                    section="company_overview",
                    excerpt="2,833 stores",
                )
            ],
            planner=_planner("vector_search"),
            status="success",
            errors=[],
        )

    override_agent_service(handler)
    response = client.post(
        "/api/v1/agent/ask",
        json={**VALID_REQUEST, "question": "How many stores did the group operate?"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["route_strategy"] == "vector_search"
    assert payload["reranked"] is True
    assert payload["executed_steps"]


def test_agent_ask_error_response_shape(client, override_agent_service):
    def handler(**_kwargs):
        return AgentAskResponse(
            session_id=VALID_REQUEST["session_id"],
            turn_index=1,
            question=VALID_REQUEST["question"],
            status="error",
            error="Synthetic failure",
            errors=["Synthetic failure"],
        )

    override_agent_service(handler)
    response = client.post("/api/v1/agent/ask", json=VALID_REQUEST)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert payload["error"] == "Synthetic failure"
    assert payload["errors"] == ["Synthetic failure"]
