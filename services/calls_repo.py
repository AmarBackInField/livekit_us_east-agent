"""CRUD operations for call records (JSON storage)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from . import json_storage


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def create_call(record: dict) -> dict:
    record.setdefault("started_at", _now())
    record.setdefault("transcript", [])
    record.setdefault("tool_calls", [])
    record.setdefault("voicemail_detected", False)
    record.setdefault("webhook_status", "pending")
    record.setdefault("end_reason", None)
    return record


async def get_call(call_id: str) -> Optional[dict]:
    return await json_storage.get_call(call_id)


async def append_transcript(call_id: str, message: dict) -> None:
    pass


async def append_tool_call(call_id: str, tool_call: dict) -> None:
    pass


async def finalize_call(
    call_id: str,
    end_reason: str,
    voicemail_detected: bool = False,
    transcript: Optional[list] = None,
    tool_calls: Optional[list] = None,
    call_data: Optional[dict] = None,
) -> Optional[dict]:
    if call_data is None:
        call_data = {"call_id": call_id}
    
    call_data["ended_at"] = _now()
    call_data["end_reason"] = end_reason
    call_data["voicemail_detected"] = voicemail_detected
    
    if transcript is not None:
        call_data["transcript"] = transcript
    if tool_calls is not None:
        call_data["tool_calls"] = tool_calls
    
    started = call_data.get("started_at")
    ended = call_data["ended_at"]
    if started and ended:
        if isinstance(started, str):
            from dateutil import parser
            started = parser.parse(started)
        duration = (ended - started).total_seconds()
        call_data["duration_sec"] = duration
    
    await json_storage.save_call(call_data)
    return call_data


async def mark_webhook_sent(call_id: str, status_code: int, error: Optional[str] = None) -> None:
    call_data = await json_storage.get_call(call_id)
    if call_data:
        status = "sent" if 200 <= status_code < 300 else "failed"
        call_data["webhook_status"] = status
        call_data["webhook_response_code"] = status_code
        if error:
            call_data["webhook_error"] = error
        await json_storage.save_call(call_data)
