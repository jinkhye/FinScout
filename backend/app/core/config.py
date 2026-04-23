from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    repository_root: Path
    backend_root: Path
    uploads_dir: Path
    pipeline_dir: Path
    logs_dir: Path
    app_name: str = "FinScout Document API"
    api_version: str = "0.1.0"
    llamaparse_tier: str = "agentic"
    llamaparse_version: str = "latest"
    llamaparse_batch_size: int = 10
    llamaparse_ignore_text_in_image: bool = True
    gemini_model: str = "gemini-3.1-flash-lite-preview"
    gemini_embedding_model: str = "gemini-embedding-2"
    gemini_embedding_delay_sec: float = 1.5
    gemini_embedding_max_retries: int = 3
    gemini_max_retries: int = 10
    gemini_retry_delay_sec: float = 1.0
    qdrant_url: str = "http://localhost:6333"
    qdrant_distance: str = "cosine"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    repository_root = Path(__file__).resolve().parents[3]
    backend_root = repository_root / "backend"
    load_dotenv(repository_root / ".env")

    return Settings(
        repository_root=repository_root,
        backend_root=backend_root,
        uploads_dir=backend_root / "storage" / "uploads",
        pipeline_dir=backend_root / "storage" / "pipelines",
        logs_dir=backend_root / "logs",
    )
