"""Append-only JSONL audit log. Only masked text is ever written."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def log_event(path: Path, event: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **event}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        # Logging must never break the chat.
        pass
