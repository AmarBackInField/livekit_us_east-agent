"""CRUD operations on the `agents` collection."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from .db import get_db


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _strip_oid(doc: dict) -> dict:
    """Remove Mongo's _id so it can be JSON-serialised cleanly."""
    if doc is None:
        return doc
    doc.pop("_id", None)
    return doc


async def create_agent(payload: dict) -> dict:
    db = get_db()
    agent_id = f"ag_{uuid.uuid4().hex[:16]}"
    doc = {
        "agent_id": agent_id,
        "created_at": _now(),
        "updated_at": _now(),
        **payload,
    }
    await db.agents.insert_one(doc)
    return _strip_oid(doc)


async def get_agent(agent_id: str) -> Optional[dict]:
    db = get_db()
    doc = await db.agents.find_one({"agent_id": agent_id})
    return _strip_oid(doc) if doc else None


async def list_agents(skip: int = 0, limit: int = 50) -> list[dict]:
    db = get_db()
    cursor = db.agents.find().sort("created_at", -1).skip(skip).limit(limit)
    return [_strip_oid(d) async for d in cursor]


async def update_agent(agent_id: str, partial: dict) -> Optional[dict]:
    db = get_db()
    if not partial:
        return await get_agent(agent_id)
    partial = {k: v for k, v in partial.items() if v is not None}
    partial["updated_at"] = _now()
    res = await db.agents.find_one_and_update(
        {"agent_id": agent_id},
        {"$set": partial},
        return_document=True,
    )
    return _strip_oid(res) if res else None


async def delete_agent(agent_id: str) -> bool:
    db = get_db()
    res = await db.agents.delete_one({"agent_id": agent_id})
    return res.deleted_count > 0


async def count_agents() -> int:
    db = get_db()
    return await db.agents.count_documents({})
