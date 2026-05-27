"""
Multi-Agent Voice Platform API.

- POST   /agents              create agent
- GET    /agents              list agents (paginated)
- GET    /agents/{id}         fetch one
- PATCH  /agents/{id}         partial update
- DELETE /agents/{id}         delete
- POST   /call                trigger outbound call (agent_id + phone_number)
- GET    /calls/{call_id}     fetch call record + transcript
- GET    /stats               RAG stats (legacy)
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from livekit import api

from models.schemas import (
    AgentCreate,
    AgentListResponse,
    AgentResponse,
    AgentUpdate,
    CallCreate,
    CallCreateResponse,
    CallResponse,
)
from services import agents_repo, calls_repo, db as db_module

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-platform")


# ---------------------------- Lifespan ----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await db_module.init_indexes()
    except Exception as e:
        logger.error("Mongo init failed: %s", e)
    yield
    await db_module.close()


app = FastAPI(
    title="Voice Agent Platform API",
    description="Create, manage, and dispatch LiveKit voice agents.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------- Helpers ----------------------------

def _serialize(doc: dict) -> dict:
    """Convert datetimes to isoformat strings for response."""
    if not doc:
        return doc
    out = dict(doc)
    for k in ("created_at", "updated_at", "started_at", "ended_at"):
        if isinstance(out.get(k), object) and hasattr(out.get(k), "isoformat"):
            out[k] = out[k].isoformat()
    return out


# ---------------------------- Root ----------------------------

@app.get("/")
async def root():
    return {
        "service": "Voice Agent Platform",
        "version": "2.0.0",
        "endpoints": {
            "POST /agents": "Create agent",
            "GET /agents": "List agents",
            "GET /agents/{agent_id}": "Get agent",
            "PATCH /agents/{agent_id}": "Update agent",
            "DELETE /agents/{agent_id}": "Delete agent",
            "POST /call": "Initiate outbound call",
            "GET /calls/{call_id}": "Fetch call record",
        },
    }


# ---------------------------- Agent CRUD ----------------------------

@app.post("/agents", response_model=AgentResponse, status_code=201)
async def create_agent(payload: AgentCreate):
    doc = await agents_repo.create_agent(payload.model_dump())
    return _serialize(doc)


@app.get("/agents", response_model=AgentListResponse)
async def list_agents(skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    items = await agents_repo.list_agents(skip=skip, limit=limit)
    total = await agents_repo.count_agents()
    return {"total": total, "agents": [_serialize(d) for d in items]}


@app.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str):
    doc = await agents_repo.get_agent(agent_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _serialize(doc)


@app.patch("/agents/{agent_id}", response_model=AgentResponse)
async def update_agent(agent_id: str, payload: AgentUpdate):
    partial = payload.model_dump(exclude_unset=True)
    doc = await agents_repo.update_agent(agent_id, partial)
    if not doc:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _serialize(doc)


@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str):
    ok = await agents_repo.delete_agent(agent_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"status": "deleted", "agent_id": agent_id}


# ---------------------------- Outbound Call ----------------------------

@app.post("/call", response_model=CallCreateResponse, status_code=201)
async def make_outbound_call(req: CallCreate):
    livekit_url = os.getenv("LIVEKIT_URL")
    livekit_api_key = os.getenv("LIVEKIT_API_KEY")
    livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")
    sip_trunk_id = os.getenv("LIVEKIT_SIP_TRUNK_ID")

    if not all([livekit_url, livekit_api_key, livekit_api_secret, sip_trunk_id]):
        raise HTTPException(status_code=500, detail="LiveKit credentials not configured")

    # Validate agent exists
    agent_doc = await agents_repo.get_agent(req.agent_id)
    if not agent_doc:
        raise HTTPException(status_code=404, detail=f"Agent {req.agent_id} not found")

    phone_number = req.phone_number.strip()
    if not phone_number.startswith("+"):
        phone_number = "+" + phone_number

    call_id = f"call_{uuid.uuid4().hex[:16]}"
    room_name = f"call-{uuid.uuid4().hex[:8]}"
    metadata = json.dumps({"agent_id": req.agent_id, "call_id": call_id})

    lk_api = api.LiveKitAPI(livekit_url, livekit_api_key, livekit_api_secret)
    try:
        # Create room first WITH metadata so the worker can read it on join
        await lk_api.room.create_room(
            api.CreateRoomRequest(name=room_name, metadata=metadata)
        )

        # Create SIP participant in that room (initiates the call)
        sip_participant = await lk_api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=sip_trunk_id,
                sip_call_to=phone_number,
                room_name=room_name,
                participant_identity=f"phone-{phone_number.replace('+', '')}",
                participant_name="Phone User",
            )
        )

        return CallCreateResponse(
            status="success",
            call_id=call_id,
            sip_call_id=sip_participant.sip_call_id,
            room_name=room_name,
            message=f"Call initiated to {phone_number}",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to initiate call")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await lk_api.aclose()


@app.get("/calls/{call_id}", response_model=CallResponse)
async def get_call(call_id: str):
    doc = await calls_repo.get_call(call_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Call not found")
    return _serialize(doc)


# ---------------------------- Legacy RAG ----------------------------

@app.get("/stats")
async def get_rag_stats():
    """Legacy RAG index stats."""
    try:
        from RAGService import RAGService
        rag = RAGService(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            index_path="./faiss_index",
        )
        return rag.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
