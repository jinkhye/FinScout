from __future__ import annotations

from textwrap import dedent
from typing import Sequence

from ...schemas.query import QueryPlanResponse
from ...schemas.vector import SectionLabel, VectorQueryResult


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
        1. Decide whether this turn is a report_question or a direct_reply.
        2. Use direct_reply for greetings, acknowledgements, casual chit-chat, or conversational turns that do not need annual-report retrieval.
        3. Use report_question for turns that need annual-report evidence.
        4. If intent=report_question, decide whether this is a single-step or multi-step retrieval problem.
        5. Set is_multi_step=true only when the user is asking for multiple distinct pieces of evidence, different report sections, or a compare/explain combination that should be retrieved separately.
        6. If intent=report_question, return the smallest useful list of sub_queries.
        7. Each sub_query must include:
           - query: a clear rewritten retrieval query for that step
           - selected_sections: the section labels for that step, or [] to search all sections
           - goal: a short description of what that step is trying to find
        8. Preserve the original financial intent. Do not add new facts.
        9. Do not split simple rephrasings or closely related facts that can be answered from one section.
        10. If intent=direct_reply, return is_multi_step=false and sub_queries=[].
        11. Use only the available section labels. Do not invent labels.
        12. Use the conversation context only to resolve references such as "that", "the same company", or "what about last year".
        13. Do not let conversation context override the current report metadata.

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


def build_retrieval_repair_prompt(
    *,
    question: str,
    planner: QueryPlanResponse,
    current_query: str,
    current_sections: Sequence[SectionLabel],
    results: Sequence[VectorQueryResult],
) -> str:
    current_sections_label = (
        ", ".join(current_sections) if current_sections else "all sections"
    )
    candidates: list[str] = []
    for index, result in enumerate(results, start=1):
        candidates.append(
            dedent(
                f"""
                Candidate {index}
                Page: {result.page_number}
                Section: {result.section}
                Has table: {result.has_table}
                Score: {result.score}
                Text excerpt:
                {result.text_for_embedding}
                """
            ).strip()
        )

    return dedent(
        f"""
        You are repairing a weak retrieval query for an annual-report RAG system.

        Report context:
        - Company: {planner.company_name}
        - Year: {planner.year}
        - Document: annual report

        Original user question:
        {question}

        Current optimized retrieval query:
        {current_query}

        Current selected sections:
        {current_sections_label}

        The first retrieval pass looked weak. Your task is to produce a better retrieval query while preserving the user's original intent.

        Instructions:
        - Rewrite the retrieval query so it is clearer and more likely to retrieve the right pages.
        - Keep the original intent. Do not add facts or change what the user is asking.
        - Keep the current section filter if it still seems appropriate.
        - Broaden the section filter only if the first retrieval likely failed because it was too narrow.
        - To search all sections, return selected_sections=[].
        - Use only these section labels when selecting sections:
          company_overview, mda, auditor_report, balance_sheet, income_statement, equitychange_statement, cashflow_statement, notes, other
        - Give a short reason for the repair.

        Weak retrieval candidates:
        {"\n\n".join(candidates) if candidates else "No candidates were returned."}
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


def build_multi_step_answer_prompt(
    *,
    question: str,
    planner: QueryPlanResponse,
    executed_steps_summary: str,
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

        Executed retrieval steps:
        {executed_steps_summary}

        Instructions:
        - Answer the full user question by synthesizing across all supplied step evidence.
        - Answer only using the supplied context. Do not use prior knowledge.
        - If the supplied context does not contain enough information to answer fully, say so explicitly.
        - Always include exact figures, percentages, and currencies when available.
        - If financial tables are in RM'000, convert figures to their true value in your answer (e.g. 100,000 in RM'000 = RM100,000,000).
        - Connect evidence across sections when helpful, but do not speculate beyond what the context states.
        - Cite the page numbers that support your answer in the citations field.
        - Do not mention retrieval internals, vector search, reranking, or step execution mechanics.
        - Use conversation context only for continuity. The supplied context for this turn remains the authoritative source.

        Supplied context:
        {context_text}
        """
    ).strip()


def build_direct_reply_prompt(
    *,
    question: str,
    company_name: str,
    year: str,
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
        You are FinScout, a conversational assistant for annual-report questions.

        Current report context:
        - Company: {company_name}
        - Year: {year}
        - Document: annual report

        User message:
        {question}
        {conversation_block}

        Instructions:
        - This turn does not require report retrieval.
        - Reply naturally, briefly, and conversationally.
        - Do not claim report facts unless retrieval was performed.
        - You may invite the user to ask about the annual report when helpful.
        - Return an empty citations list.
        """
    ).strip()
