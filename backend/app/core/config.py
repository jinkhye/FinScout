from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Settings:
    repository_root: Path
    backend_root: Path
    uploads_dir: Path
    pipeline_dir: Path
    pipeline_config_path: Path
    app_name: str = "FinScout Document API"
    api_version: str = "0.1.0"
    llamaparse_tier: str = "agentic"
    llamaparse_version: str = "latest"
    llamaparse_batch_size: int = 10
    llamaparse_ignore_text_in_image: bool = True
    gemini_model: str = "gemini-3.1-flash-lite-preview"
    gemini_max_retries: int = 10
    gemini_retry_delay_sec: float = 1.0


def _resolve_path(root: Path, raw_value: Any, default: Path) -> Path:
    if raw_value is None:
        return default

    candidate = Path(str(raw_value))
    if candidate.is_absolute():
        return candidate
    return root / candidate


def _load_pipeline_config(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}

    with path.open("rb") as handle:
        payload = tomllib.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Pipeline config must be a TOML table: {path}")
    return payload


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    repository_root = Path(__file__).resolve().parents[3]
    backend_root = repository_root / "backend"
    pipeline_config_path = backend_root / "pipeline.toml"
    pipeline_config = _load_pipeline_config(pipeline_config_path)

    storage_config = pipeline_config.get("storage", {})
    llamaparse_config = pipeline_config.get("llamaparse", {})
    gemini_config = pipeline_config.get("gemini", {})

    if not isinstance(storage_config, dict):
        storage_config = {}
    if not isinstance(llamaparse_config, dict):
        llamaparse_config = {}
    if not isinstance(gemini_config, dict):
        gemini_config = {}

    return Settings(
        repository_root=repository_root,
        backend_root=backend_root,
        uploads_dir=_resolve_path(
            repository_root,
            storage_config.get("uploads_dir"),
            backend_root / "storage" / "uploads",
        ),
        pipeline_dir=_resolve_path(
            repository_root,
            storage_config.get("pipeline_dir"),
            backend_root / "storage" / "pipelines",
        ),
        pipeline_config_path=pipeline_config_path,
        llamaparse_tier=str(llamaparse_config.get("tier", "agentic")),
        llamaparse_version=str(llamaparse_config.get("version", "latest")),
        llamaparse_batch_size=int(llamaparse_config.get("batch_size", 10)),
        llamaparse_ignore_text_in_image=bool(
            llamaparse_config.get("ignore_text_in_image", True)
        ),
        gemini_model=str(gemini_config.get("model", "gemini-3.1-flash-lite-preview")),
        gemini_max_retries=int(gemini_config.get("max_retries", 10)),
        gemini_retry_delay_sec=float(gemini_config.get("retry_delay_sec", 1.0)),
    )
