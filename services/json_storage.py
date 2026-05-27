"""JSON file storage for call records."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

CALL_LOGS_DIR = Path("call_logs")


def _ensure_dir() -> None:
    """Ensure the call_logs directory exists."""
    CALL_LOGS_DIR.mkdir(exist_ok=True)


def _get_call_path(call_id: str) -> Path:
    """Get the file path for a call's JSON file."""
    return CALL_LOGS_DIR / f"call_{call_id}.json"


def _serialize_datetime(obj):
    """JSON serializer for datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


async def save_call(call_data: dict) -> None:
    """Save call data to a JSON file."""
    _ensure_dir()
    call_id = call_data.get("call_id")
    if not call_id:
        raise ValueError("call_data must contain 'call_id'")
    
    file_path = _get_call_path(call_id)
    
    # Write to file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(call_data, f, indent=2, default=_serialize_datetime)


async def get_call(call_id: str) -> Optional[dict]:
    """Load call data from a JSON file."""
    file_path = _get_call_path(call_id)
    
    if not file_path.exists():
        return None
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


async def list_calls(limit: int = 50) -> list[dict]:
    """List recent call files (sorted by modification time)."""
    _ensure_dir()
    
    call_files = sorted(
        CALL_LOGS_DIR.glob("call_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )[:limit]
    
    calls = []
    for file_path in call_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                calls.append(json.load(f))
        except Exception:
            continue
    
    return calls
