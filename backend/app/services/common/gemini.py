from __future__ import annotations

import os
from typing import Any

from google import genai


def get_gemini_client() -> Any | None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)
