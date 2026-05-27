"""MongoDB connection management (async, via Motor)."""
from __future__ import annotations

import os
import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING

logger = logging.getLogger(__name__)

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        uri = os.getenv("MONGODB_URI")
        if not uri:
            raise RuntimeError("MONGODB_URI is not set")
        _client = AsyncIOMotorClient(uri, maxPoolSize=100, serverSelectionTimeoutMS=5000)
    return _client


def get_db() -> AsyncIOMotorDatabase:
    global _db
    if _db is None:
        db_name = os.getenv("DB_NAME", "Cartesia")
        _db = get_client()[db_name]
    return _db


async def init_indexes() -> None:
    """Idempotent index creation. Call on FastAPI startup."""
    db = get_db()
    await db.agents.create_index([("agent_id", ASCENDING)], unique=True)
    logger.info("MongoDB indexes ensured")


async def close() -> None:
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None
