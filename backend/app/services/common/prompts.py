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
    conversation_context: str = "",
) -> str:
    sections = ", ".join(available_sections)
    conversation_block = ""
    if conversation_context.strip():
        conversation_block = dedent(
            f"""

            Conversation context:
            {conversation_context}
            """
        )
    return dedent(
        f"""
        You are the query planner for an annual-report RAG system.

        Report context:
        - Company: {company_name}
        - Year: {year}
        - Document: annual report

        User query:
        {query}
        {conversation_block}

        Available section labels:
        {sections}

        Task:
        1. Correct typos and rewrite the user query as a clear retrieval query.
        2. Preserve the original financial intent. Do not add new facts.
        3. Decide whether the query should search all sections or specific sections.
        4. If the query is broad, unclear, or spans the whole annual report, return selected_sections=[].
        5. If the query clearly maps to one or more section labels, return only those labels in selected_sections.
        6. Use only the available section labels. Do not invent labels.
        7. Use the conversation context only to resolve references such as "that", "the same company", or "what about last year".
        8. Do not let conversation context override the current report metadata.

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
        excerpt = result.text_for_embedding
        candidates.append(
            dedent(
                f"""
                Candidate {index}
                Page: {result.page_number}
                Section: {result.section}
                Has table: {result.has_table}
                Score: {result.score}
                Text excerpt:
                {excerpt}
                """
            ).strip()
        )

    return dedent(
        f"""
        You are a relevance judge for an annual-report RAG system.

        Company: {planner.company_name}
        Year: {planner.year}
        Original question: {question}
        Optimized retrieval query: {planner.optimized_query}

        Your task is to select the best 3 to 5 candidates that directly answer the question.

        Ranking criteria:
        1. Direct relevance — does this page directly contain the answer?
        2. Specificity — does it contain concrete financial figures, names, or facts rather than vague narrative?
        3. Recency — if multiple years are present, prefer the most recent data unless the question asks for historical comparison.

        Penalize pages that are only tangentially related or contain no specific answer to the question.
        If a candidate has tables, its text excerpt already includes the table content summarized into plain language.

        Candidates:
        {"\n\n".join(candidates)}
        """
    ).strip()


def build_answer_prompt(
    *,
    question: str,
    planner: QueryPlanResponse,
    context_text: str,
    conversation_context: str = "",
) -> str:
    conversation_block = ""
    if conversation_context.strip():
        conversation_block = dedent(
            f"""

            Conversation context:
            {conversation_context}
            """
        )
    return dedent(
        f"""
        You are FinScout, an annual-report question answering assistant.

        Report context:
        - Company: {planner.company_name}
        - Year: {planner.year}
        - Document: annual report

        User question:
        {question}
        {conversation_block}

        Instructions:
        - Answer only using the supplied context. Do not use prior knowledge.
        - If the context does not contain enough information to answer, say so explicitly.
        - Always include exact figures, percentages, and currencies when available.
        - If financial tables are in RM'000, convert figures to their true value in your answer (e.g. 100,000 in RM'000 = RM100,000,000).
        - Cite the page numbers that support your answer in the citations field.
        - Do not mention retrieval internals, vector search, or reranking.
        - Do not speculate beyond what the context states.
        - Use conversation context only for continuity. The supplied context for this turn remains the authoritative source.

        Supplied context:
        {context_text}
        """
    ).strip()
