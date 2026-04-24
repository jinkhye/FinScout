from __future__ import annotations

import os
import time
from logging import Logger
from typing import Any, List

from google import genai


def get_gemini_client() -> Any | None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def embed_text_with_retries(
    *,
    gemini_client: Any,
    model: str,
    text: str,
    max_retries: int,
    retry_delay_sec: float,
    logger: Logger,
    label: str,
) -> List[float]:
    for attempt in range(max_retries + 1):
        try:
            response = gemini_client.models.embed_content(model=model, contents=text)
            return extract_embedding(response)
        except Exception as exc:
            if attempt >= max_retries:
                raise

            wait_seconds = embedding_retry_delay(exc, attempt, retry_delay_sec)
            logger.warning(
                "Embedding %s failed, retrying in %.1fs: %s",
                label,
                wait_seconds,
                exc,
            )
            time.sleep(wait_seconds)

    raise ValueError(f"Embedding {label} failed")


def extract_embedding(response: Any) -> List[float]:
    embeddings = getattr(response, "embeddings", None)
    if embeddings:
        values = getattr(embeddings[0], "values", None)
        if values:
            return [float(value) for value in values]

    raise ValueError("Gemini embedding response did not contain vector values")


def embedding_retry_delay(
    exc: Exception, attempt: int, retry_delay_sec: float
) -> float:
    message = str(exc).lower()
    if "429" in message or "resource_exhausted" in message or "quota" in message:
        return 60.0
    return retry_delay_sec * (2**attempt)
