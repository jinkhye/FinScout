from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List

from ...core.config import Settings


@dataclass(frozen=True)
class ConversationSession:
    session_id: str
    processed_file_path: str
    collection_name: str
    company_name: str
    year: str


@dataclass(frozen=True)
class ConversationTurn:
    turn_index: int
    question: str
    intent: str
    optimized_query: str
    selected_sections: list[str]
    route_strategy: str
    reranked: bool
    answer: str
    citations: list[dict[str, Any]]
    created_at: str


class ConversationMemoryService:
    def __init__(self, settings: Settings) -> None:
        self._db_path = settings.conversations_db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def get_or_create_session(
        self,
        *,
        session_id: str,
        processed_file_path: str,
        collection_name: str,
        company_name: str,
        year: str,
    ) -> ConversationSession:
        now = self._timestamp()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT session_id, processed_file_path, collection_name, company_name, year
                FROM conversation_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

            if row is None:
                connection.execute(
                    """
                    INSERT INTO conversation_sessions (
                        session_id,
                        processed_file_path,
                        collection_name,
                        company_name,
                        year,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        processed_file_path,
                        collection_name,
                        company_name,
                        year,
                        now,
                        now,
                    ),
                )
                return ConversationSession(
                    session_id=session_id,
                    processed_file_path=processed_file_path,
                    collection_name=collection_name,
                    company_name=company_name,
                    year=year,
                )

            if row["processed_file_path"] != processed_file_path:
                raise ValueError(
                    "session_id is already tied to a different processed_file_path"
                )
            if row["collection_name"] != collection_name:
                raise ValueError(
                    "session_id is already tied to a different collection_name"
                )

            connection.execute(
                """
                UPDATE conversation_sessions
                SET updated_at = ?
                WHERE session_id = ?
                """,
                (now, session_id),
            )
            return ConversationSession(
                session_id=str(row["session_id"]),
                processed_file_path=str(row["processed_file_path"]),
                collection_name=str(row["collection_name"]),
                company_name=str(row["company_name"]),
                year=str(row["year"]),
            )

    def list_recent_turns(
        self,
        session_id: str,
        limit: int = 3,
    ) -> List[ConversationTurn]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    turn_index,
                    question,
                    intent,
                    optimized_query,
                    selected_sections,
                    route_strategy,
                    reranked,
                    answer,
                    citations,
                    created_at
                FROM conversation_turns
                WHERE session_id = ?
                ORDER BY turn_index DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        turns = [self._row_to_turn(row) for row in reversed(rows)]
        return turns

    def next_turn_index(self, session_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(MAX(turn_index), 0) AS max_turn_index
                FROM conversation_turns
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        max_turn_index = int(row["max_turn_index"]) if row is not None else 0
        return max_turn_index + 1

    def append_turn(
        self,
        *,
        session_id: str,
        turn_index: int,
        question: str,
        intent: str,
        optimized_query: str,
        selected_sections: list[str],
        route_strategy: str,
        reranked: bool,
        answer: str,
        citations: list[dict[str, Any]],
    ) -> None:
        now = self._timestamp()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversation_turns (
                    session_id,
                    turn_index,
                    question,
                    intent,
                    optimized_query,
                    selected_sections,
                    route_strategy,
                    reranked,
                    answer,
                    citations,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    turn_index,
                    question,
                    intent,
                    optimized_query,
                    json.dumps(selected_sections, ensure_ascii=False),
                    route_strategy,
                    1 if reranked else 0,
                    answer,
                    json.dumps(citations, ensure_ascii=False),
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE conversation_sessions
                SET updated_at = ?
                WHERE session_id = ?
                """,
                (now, session_id),
            )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_sessions (
                    session_id TEXT PRIMARY KEY,
                    processed_file_path TEXT NOT NULL,
                    collection_name TEXT NOT NULL,
                    company_name TEXT NOT NULL,
                    year TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    intent TEXT NOT NULL DEFAULT 'report_question',
                    optimized_query TEXT NOT NULL,
                    selected_sections TEXT NOT NULL,
                    route_strategy TEXT NOT NULL,
                    reranked INTEGER NOT NULL,
                    answer TEXT NOT NULL,
                    citations TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, turn_index),
                    FOREIGN KEY (session_id) REFERENCES conversation_sessions(session_id)
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(conversation_turns)"
                ).fetchall()
            }
            if "intent" not in columns:
                connection.execute(
                    """
                    ALTER TABLE conversation_turns
                    ADD COLUMN intent TEXT NOT NULL DEFAULT 'report_question'
                    """
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _row_to_turn(self, row: sqlite3.Row) -> ConversationTurn:
        selected_sections = json.loads(str(row["selected_sections"]) or "[]")
        citations = json.loads(str(row["citations"]) or "[]")
        return ConversationTurn(
            turn_index=int(row["turn_index"]),
            question=str(row["question"]),
            intent=str(row["intent"]),
            optimized_query=str(row["optimized_query"]),
            selected_sections=[
                str(section) for section in selected_sections if isinstance(section, str)
            ],
            route_strategy=str(row["route_strategy"]),
            reranked=bool(row["reranked"]),
            answer=str(row["answer"]),
            citations=[item for item in citations if isinstance(item, dict)],
            created_at=str(row["created_at"]),
        )

    def _timestamp(self) -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")
