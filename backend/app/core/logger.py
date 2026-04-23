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


def _run_directory(logs_dir: Path, endpoint_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_endpoint = endpoint_name.strip("/").replace("/", "_").replace("-", "_")
    return logs_dir / safe_endpoint / timestamp
