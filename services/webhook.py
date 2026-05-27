"""Outbound webhook delivery (post-call payload)."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} is not JSON serializable")


async def send_webhook(url: str, payload: dict, timeout: float = 10.0) -> tuple[int, str | None]:
    """Returns (status_code, error_message). status_code=0 means transport failure."""
    if not url:
        return 0, "no webhook url"
    try:
        body = json.dumps(payload, default=_json_default)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                url,
                content=body,
                headers={"Content-Type": "application/json"},
            )
            return resp.status_code, None
    except Exception as e:
        logger.exception("Webhook delivery failed: %s", e)
        return 0, str(e)
