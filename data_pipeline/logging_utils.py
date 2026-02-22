from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def write_jsonl_record(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_event(event: str, **extra) -> dict:
    return {
        "event": event,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
