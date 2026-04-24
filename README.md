# FinScout

FinScout is an annual-report question answering system built around retrieval-augmented generation for Malaysian public company reports.

It combines:
- a FastAPI backend for planning, retrieval, answer generation, and evaluation
- a Next.js frontend for chat + PDF review
- a report pipeline that stores processed report text and vector index metadata

The current demo is centered on the 2024 annual report for **99 Speed Mart Retail Holdings Berhad**.

## What FinScout does

FinScout answers questions about an annual report by:
- understanding the user query
- deciding whether it needs retrieval at all
- choosing between full-context loading and vector retrieval
- reranking vector results
- retrying retrieval once if the first retrieval looks weak
- synthesizing a grounded answer with citations

It also supports:
- conversation memory across turns
- multi-step planning for more complex questions
- PDF-linked citations in the frontend
- backend golden evaluation using Gemini as judge

## Repo structure

```text
FinScout/
|- backend/
|  |- app/
|  |  |- api/
|  |  |- core/
|  |  |- evaluation/
|  |  |- schemas/
|  |  `- services/
|  |- examples/
|  |- logs/
|  |- storage/
|  `- tests/
|- frontend/
|  |- app/
|  |- components/
|  `- public/
`- README.md
```

## Core backend flow

The main user-facing endpoint is:

```text
POST /api/v1/agent/ask
```

High-level flow:

```text
user query
-> conversation memory
-> query planner
-> direct_reply OR report_question
-> for report questions: one or more planned sub-queries
-> execute up to 2 steps
-> full_context OR vector_search per step
-> for vector_search: retrieve -> rerank -> optional repair/retry -> rerank
-> final synthesized answer
-> citations
-> save turn to memory
```

Important supporting routes:
- `/api/v1/agent/ask`
- `/api/v1/query/plan`
- `/api/v1/query/context`
- `/api/v1/vector/query`
- `/api/v1/vector/ingest`
- `/api/v1/documents/report`
- `/api/v1/documents/report/pdf`
- `/health`

## Tech stack

### Backend
- FastAPI
- Pydantic
- Google Gemini API
- Qdrant
- SQLite for conversation memory

### Frontend
- Next.js 14
- React 18
- `react-pdf`

## Local setup

## 1. Backend

From repo root:

```powershell
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

Run the backend:

```powershell
.venv\Scripts\python.exe -m backend.app.main
```

Backend default URL:

```text
http://127.0.0.1:8000
```

## 2. Frontend

Install frontend dependencies:

```powershell
cd frontend
npm install
```

Run the frontend:

```powershell
npm run dev
```

Frontend default URL:

```text
http://127.0.0.1:3000
```

## Environment and services

FinScout expects:
- a valid Gemini API key in `.env`
- a running Qdrant instance at `http://localhost:6333` by default

The backend settings live in [config.py](backend/app/core/config.py).

Current defaults include:
- Gemini model: `gemini-3.1-flash-lite-preview`
- embedding model: `gemini-embedding-2`
- Qdrant URL: `http://localhost:6333`

## Demo report

The current demo uses:

- processed pipeline file:
  - `backend/storage/pipelines/99SMART-Annual-Report-2024/processed_99SMART-Annual-Report-2024.json`
- uploaded PDF:
  - `backend/storage/uploads/99SMART-Annual-Report-2024.pdf`

The frontend is currently wired as a single-report demo around this report.

## Logs and artifacts

The backend writes run artifacts under:

- `backend/logs/agent_ask/`
- `backend/logs/agent_ask_eval/`
- `backend/logs/vector_ingest/`

These are useful for debugging:
- request payloads
- planner outputs
- retrieval results
- reranker outputs
- answer prompts
- evaluation summaries

## Testing and evaluation

There are two backend quality layers:

1. deterministic endpoint tests for `/api/v1/agent/ask`
2. golden evaluation with Gemini as judge

See [TEST_CASES.md](TEST_CASES.md) for the full workflow.

Quick commands:

```powershell
python -m pytest backend\tests\test_agent_ask_api.py -q
python -m backend.app.evaluation.agent_ask_golden_eval
```

## Current notes

- The backend evaluation dataset is intentionally small and demo-focused.
- The frontend is optimized for the 99 Speed Mart demo report rather than a full multi-report UX.
- Multi-step planning and execution are implemented, with execution capped for demo safety.

## Project status

FinScout is currently strongest as:
- a report-grounded RAG demo
- a retrieval/planning playground
- a chat + PDF inspection tool for annual-report QA

The repo already includes enough infrastructure to keep improving:
- routing logic
- retrieval quality
- answer grounding
- evaluation discipline
- frontend transparency
