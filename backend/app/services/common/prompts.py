from __future__ import annotations

from textwrap import dedent
from typing import Sequence

from ...schemas.query import QueryPlanResponse
from ...schemas.vector import VectorQueryResult


def build_query_planner_prompt(
    *,
    query: str,
    company_name: str,
    year: str,
    available_sections: Sequence[str],
) -> str:
    sections = ", ".join(available_sections)
    return dedent(
        f"""
        You are the query planner for an annual-report RAG system.

        Report context:
        - Company: {company_name}
        - Year: {year}
        - Document: annual report

        User query:
        {query}

        Available section labels:
        {sections}

        Task:
        1. Correct typos and rewrite the user query as a clear retrieval query.
        2. Preserve the original financial intent. Do not add new facts.
        3. Decide whether the query should search all sections or specific sections.
        4. If the query is broad, unclear, or spans the whole annual report, set no_filter=true and selected_sections=[].
        5. If the query clearly maps to one or more section labels, set no_filter=false and selected_sections to those labels only.
        6. Use only the available section labels. Do not invent labels.
        7. If selected_sections contains only financial statement sections (balance_sheet, income_statement, equitychange_statement, cashflow_statement), set direct_load=true. Otherwise set direct_load=false.

        Section guidance:
        - company_overview: corporate profile, milestones, directors, business overview, outlets, strategy
        - mda: chairman statement, MD review, management discussion, operations, performance narrative
        - auditor_report: audit opinion, auditor, key audit matters, audit responsibilities
        - balance_sheet: assets, liabilities, equity, financial position
        - income_statement: revenue, profit, EPS, income, comprehensive income
        - equitychange_statement: dividends, reserves, issued capital, changes in equity
        - cashflow_statement: operating, investing, financing cash flows, cash balances
        - notes: accounting policies, detailed notes, segment details, commitments, related parties
        - other: anything that does not fit the above
        """
    ).strip()


def build_reranker_prompt(
    *,
    question: str,
    planner: QueryPlanResponse,
    results: Sequence[VectorQueryResult],
) -> str:
    candidates: list[str] = []
    for index, result in enumerate(results, start=1):
        excerpt = (result.text or result.text_for_embedding)[:1600]
        candidates.append(
            dedent(
                f"""
                Candidate {index}
                Page: {result.page_number}
                Section: {result.section}
                Score: {result.score}
                Text excerpt:
                {excerpt}
                """
            ).strip()
        )

    return dedent(
        f"""
        Rerank annual-report evidence pages for answering the user question.

        Company: {planner.company_name}
        Year: {planner.year}
        Original question: {question}
        Optimized query: {planner.optimized_query}

        Select the best 3 to 5 pages. Prefer pages that directly contain the answer.
        Return JSON only.

        Candidates:
        {"\n\n".join(candidates)}
        """
    ).strip()


def build_answer_prompt(
    *,
    question: str,
    planner: QueryPlanResponse,
    context_text: str,
) -> str:
    return dedent(
        f"""
        You are FinScout, an annual-report question answering assistant.

        Report context:
        - Company: {planner.company_name}
        - Year: {planner.year}
        - Document: annual report

        User question:
        {question}

        Optimized query:
        {planner.optimized_query}

        Instructions:
        - Answer only using the supplied context.
        - If the context does not contain enough information, say so.
        - Cite supporting page numbers in the citations field.
        - Keep the answer concise but include exact figures when available.
        - Do not mention retrieval internals.

        Supplied context:
        {context_text}
        """
    ).strip()
