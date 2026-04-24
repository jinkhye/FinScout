# Test Cases and Evaluation

This document explains the current backend quality workflow for FinScout.

The project uses two complementary layers:

1. deterministic API tests for `/api/v1/agent/ask`
2. golden evaluation with Gemini as judge

The goal is to catch both:
- structural regressions
- answer quality / grounding regressions

## 1. Deterministic backend tests

These tests live in:

- [conftest.py](backend/tests/conftest.py)
- [test_agent_ask_api.py](backend/tests/test_agent_ask_api.py)

They do **not** hit live Gemini or Qdrant. Instead, they override the agent dependency and verify endpoint behavior.

### What they cover

#### Request validation
- blank `session_id`
- blank `processed_file_path`
- blank `question`
- invalid `top_k`

#### Success response shape
- `direct_reply` response
- `full_context` response
- `vector_search` response

#### Error-path stability
- stable `status="error"` serialization

### Run command

From repo root:

```powershell
python -m pytest backend\tests\test_agent_ask_api.py -q
```

## 2. Golden evaluation

Golden evaluation exercises the real `/api/v1/agent/ask` flow and then asks Gemini to judge the result against expected facts, citations, and route quality.

### Main files

- dataset:
  - [agent_ask_golden_dataset.json](backend/examples/agent_ask_golden_dataset.json)
- runner:
  - [agent_ask_golden_eval.py](backend/app/evaluation/agent_ask_golden_eval.py)
- judge prompt:
  - [prompts.py](backend/app/services/common/prompts.py)

### What the judge looks at

For each case:
- question
- expected route strategy
- expected pages
- expected facts
- actual answer
- actual citations
- actual executed steps

The judge scores:
- correctness
- grounding
- citation quality
- route quality
- hallucination risk

### Judge output shape

Per-case judge output includes:
- `pass`
- `score`
- `correctness`
- `grounding`
- `citation_quality`
- `route_quality`
- `hallucination`
- `reason`
- `failures`

### Run command

From repo root:

```powershell
python -m backend.app.evaluation.agent_ask_golden_eval
```

Optional:

```powershell
python -m backend.app.evaluation.agent_ask_golden_eval --limit 5
```

### Output location

Results are written under:

```text
backend/logs/agent_ask_eval/<timestamp>/
```

Important files:
- `summary.json`
- `summary.md`
- `cases/<case_id>/result.json`
- `cases/<case_id>/judge_prompt.md`

## 3. Golden dataset design

The golden dataset currently covers:
- exact numeric financial questions
- auditor questions
- management / narrative questions
- multi-step synthesis questions
- direct-reply small talk
- not-found / insufficient-context questions

### Example covered question types

#### Fact lookup
- revenue in 2024
- profit for the financial year
- total assets
- net cash from operating activities

#### Narrative retrieval
- what management attributes growth to
- how many stores the group operated

#### Multi-step
- how profitability changed and what management attributes it to

#### Direct reply
- `hello`
- `thanks`

#### Not found / don't invent
- 2030 carbon emissions reduction target
- CFO salary in 2024
- 2025 revenue forecast

These not-found cases are especially important because they test whether the system avoids hallucinating unsupported facts.

## 4. Bootstrapping a draft golden dataset

There is also a helper to generate a **draft** dataset by calling the live endpoint.

Files:
- [bootstrap_golden_dataset.py](backend/app/evaluation/bootstrap_golden_dataset.py)
- [agent_ask_seed_questions.json](backend/examples/agent_ask_seed_questions.json)

### What it does

It sends each seed question to the live backend and creates a draft dataset using:
- returned answer
- returned citations
- returned route strategy
- returned executed steps

This is useful for creating a first pass, but it should not be treated as final gold without manual review.

### Run command

Make sure the backend server is already running, then:

```powershell
python -m backend.app.evaluation.bootstrap_golden_dataset
```

Optional:

```powershell
python -m backend.app.evaluation.bootstrap_golden_dataset --base-url http://127.0.0.1:8000
```

Default output:

```text
backend/examples/agent_ask_golden_dataset.draft.json
```

## 5. How to review bad results

When a case fails:

1. open `summary.md`
2. open the failed case's `result.json`
3. compare the answer against the processed report / cited pages
4. decide whether:
   - the model was wrong
   - the citations were weak
   - the gold expectation was wrong

This matters because a first-pass golden dataset can itself be noisy.

## 6. Recommended workflow

### Fast local safety check

```powershell
python -m pytest backend\tests\test_agent_ask_api.py -q
```

### Quality regression check

```powershell
python -m backend.app.evaluation.agent_ask_golden_eval
```

### When adding new cases

1. add seed questions if useful
2. bootstrap a draft if needed
3. manually verify facts against the report
4. update `agent_ask_golden_dataset.json`
5. rerun the eval

## 7. Current philosophy

This quality setup is intentionally pragmatic:

- deterministic tests protect endpoint behavior
- Gemini judge helps score answer quality realistically
- the dataset is small but high-signal
- unsupported-question cases are included to test grounding discipline

The aim is not perfect benchmarking yet. The aim is to make regressions visible and debugging easier while the product is still evolving.
