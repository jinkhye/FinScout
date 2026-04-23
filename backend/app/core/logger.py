from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Settings


def create_run_logger(
    settings: Settings,
    endpoint_name: str,
    request_payload: dict[str, Any],
) -> logging.Logger:
    run_dir = _run_directory(settings.logs_dir, endpoint_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "request.json").write_text(
        json.dumps(request_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger = logging.getLogger(f"finscout.{endpoint_name}.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    logger.run_dir = run_dir  # type: ignore[attr-defined]

    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.info("Run started: %s", run_dir)
    return logger


def log_json_artifact(
    logger: logging.Logger,
    relative_path: str,
    payload: Any,
) -> Path:
    path = _artifact_path(logger, relative_path)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def log_text_artifact(
    logger: logging.Logger,
    relative_path: str,
    text: str,
) -> Path:
    path = _artifact_path(logger, relative_path)
    path.write_text(text, encoding="utf-8")
    return path


def _artifact_path(logger: logging.Logger, relative_path: str) -> Path:
    run_dir = getattr(logger, "run_dir", None)
    if not isinstance(run_dir, Path):
        raise ValueError("Logger is missing run_dir")

    path = run_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _run_directory(logs_dir: Path, endpoint_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_endpoint = endpoint_name.strip("/").replace("/", "_").replace("-", "_")
    return logs_dir / safe_endpoint / timestamp
