"""
Dynamic LiveKit voice agent worker.

On each call:
  1. Reads `agent_id` + `call_id` from room metadata
  2. Loads the agent config from MongoDB
  3. Builds a dynamic Agent with the configured tools (end_call,
     voicemail_detection, language_detection, human_transfer)
  4. Streams transcript into MongoDB
  5. On call end, fires the post-call webhook with the full transcript

Pre-warmed Silero VAD is shared across processes via num_idle_processes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from cachetools import TTLCache
from dotenv import load_dotenv
from livekit import api
from livekit.agents import (
    cli,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    RoomInputOptions,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent
from livekit.plugins import cartesia, deepgram, openai, silero

# Turn-detector (LLM-based end-of-utterance predictor) — huge endpointing speedup.
# Falls back gracefully if the plugin isn't installed.
try:
    from livekit.plugins.turn_detector.english import EnglishModel as _TurnDetectorEN
    _TURN_DETECTOR_AVAILABLE = True
except Exception:  # pragma: no cover
    _TurnDetectorEN = None  # type: ignore
    _TURN_DETECTOR_AVAILABLE = False

from services import agents_repo, calls_repo, webhook
from streaming_logger import StreamingDebugLogger
from config import ProductionConfig, get_balanced_config
from network_monitor import NetworkMonitor
from redis_manager import get_redis_manager
from llm_factory import create_llm, get_provider_info

load_dotenv()

# Load production configuration
PRODUCTION_CONFIG = get_balanced_config()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("livekit-agent")

# Per-worker LRU cache for agent docs (60s TTL) so repeat calls to the
# same agent_id don't hit Mongo.
_AGENT_CACHE: TTLCache = TTLCache(maxsize=512, ttl=60)

# Voicemail detection heuristic — match phrases commonly used in cellular voicemail greetings.
VOICEMAIL_PATTERNS = [
    r"\bvoice\s*mail\b",
    r"\bleave (?:a|your) message\b",
    r"\bat the tone\b",
    r"\bplease record (?:your )?message\b",
    r"\bnot available\b.*\b(?:right now|at the moment)\b",
    r"\bcall has been forwarded\b",
    r"\bafter the (?:beep|tone)\b",
]
VOICEMAIL_RE = re.compile("|".join(VOICEMAIL_PATTERNS), re.IGNORECASE)
VOICEMAIL_DETECT_WINDOW_SEC = 8.0  # only consider voicemail within first N seconds


# ---------------------------------------------------------------------------
# Prewarm
# ---------------------------------------------------------------------------

def prewarm(proc: JobProcess) -> None:
    """Preload Silero VAD + turn-detector once per worker process."""
    proc.userdata["vad"] = silero.VAD.load(
        min_speech_duration=0.05,
        min_silence_duration=0.05,   # turn_detector handles real endpointing
        activation_threshold=0.5,
    )
    if _TURN_DETECTOR_AVAILABLE:
        try:
            proc.userdata["turn_detector"] = _TurnDetectorEN()
            logger.info("Turn detector prewarmed (LiveKit EnglishModel)")
        except Exception as e:
            logger.warning("Turn detector load failed: %s", e)
            proc.userdata["turn_detector"] = None
    else:
        proc.userdata["turn_detector"] = None
        logger.info("livekit-plugins-turn-detector not installed; using VAD-only endpointing")
    logger.info("VAD prewarmed (min_speech=0.05s, min_silence=0.05s)")


# ---------------------------------------------------------------------------
# Agent class with dynamic tools
# ---------------------------------------------------------------------------

class DynamicAgent(Agent):
    """Agent whose tool set is determined per-call from MongoDB."""

    def __init__(
        self,
        *,
        agent_doc: dict,
        ctx: JobContext,
        call_state: "CallState",
    ):
        self._agent_doc = agent_doc
        self._ctx = ctx
        self._state = call_state
        # Append brevity rule for low-latency phone responses
        instructions = (
            agent_doc["system_prompt"].rstrip()
            + "\n\nIMPORTANT: This is a phone call. Reply in 1 short sentence "
            "(max 20 words). Be concise and conversational."
        )
        super().__init__(instructions=instructions)

    # ---- end_call ----
    @function_tool()
    async def end_call(self) -> str:
        """End the current phone call. Use ONLY when the user clearly says
        goodbye, asks to hang up, or the conversation is naturally over."""
        if not self._agent_doc["tools"].get("end_call", {}).get("enabled"):
            return "end_call tool is not enabled for this agent."

        logger.info("Tool: end_call invoked")
        self._state.end_reason = "completed"
        self._state.tool_calls.append({
            "tool": "end_call",
            "at": datetime.now(timezone.utc).isoformat(),
        })
        # Speak a short closing line then hang up
        try:
            await self._state.session.say(
                "Thank you for the call. Goodbye!", allow_interruptions=False
            )
        except Exception:
            pass
        asyncio.create_task(self._hangup())
        return "Ending the call now."

    async def _hangup(self) -> None:
        await asyncio.sleep(0.5)
        try:
            await self._ctx.delete_room()
        except Exception as e:
            logger.warning("delete_room failed: %s", e)

    # ---- language_detection / switch_language ----
    @function_tool()
    async def switch_language(self, language: str) -> str:
        """Switch the assistant's TTS+STT language when the user speaks
        another language. `language` should be a 2-letter ISO code like
        'en', 'es', 'fr', 'hi'."""
        if not self._agent_doc["tools"].get("language_detection", {}).get("enabled"):
            return "language_detection tool is not enabled."

        logger.info("Tool: switch_language -> %s", language)
        self._state.tool_calls.append({
            "tool": "switch_language",
            "args": {"language": language},
            "at": datetime.now(timezone.utc).isoformat(),
        })
        try:
            self._state.session.tts.update_options(language=language)
            self._state.session.stt.update_options(language=language)
        except Exception as e:
            logger.warning("update_options failed: %s", e)
            return f"Could not switch language: {e}"
        return f"Language switched to {language}."

    # ---- human_transfer ----
    @function_tool()
    async def transfer_to_human(self, reason: str) -> str:
        """Transfer the caller to a human agent via SIP REFER. Use this
        only when the user explicitly asks for a human/supervisor or the
        request is outside your capabilities. `reason` is a short summary
        of why the transfer is needed."""
        ht = self._agent_doc["tools"].get("human_transfer", {})
        if not ht.get("enabled"):
            return "human_transfer tool is not enabled."
        rules = ht.get("rules") or []
        if not rules:
            return "No transfer rules configured."

        target = rules[0]["phone_number"]
        logger.info("Tool: transfer_to_human -> %s (reason=%s)", target, reason)
        self._state.tool_calls.append({
            "tool": "transfer_to_human",
            "args": {"reason": reason, "phone_number": target},
            "at": datetime.now(timezone.utc).isoformat(),
        })
        self._state.end_reason = "transferred"

        # Find the SIP participant identity
        sip_identity = self._state.sip_participant_identity
        if not sip_identity:
            return "Could not find SIP participant to transfer."

        try:
            await self._state.session.say(
                "Transferring you to a human agent now. Please hold.",
                allow_interruptions=False,
            )
            await self._ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    participant_identity=sip_identity,
                    room_name=self._ctx.room.name,
                    transfer_to=f"tel:{target}",
                )
            )
            return "Transfer initiated."
        except Exception as e:
            logger.exception("SIP transfer failed")
            return f"Transfer failed: {e}"


# ---------------------------------------------------------------------------
# Per-call mutable state
# ---------------------------------------------------------------------------

class CallState:
    def __init__(self, call_id: str, agent_id: str):
        self.call_id = call_id
        self.agent_id = agent_id
        self.transcript: list[dict] = []
        self.tool_calls: list[dict] = []
        self.voicemail_detected = False
        self.end_reason: Optional[str] = None
        self.sip_participant_identity: Optional[str] = None
        self.started_at: datetime = datetime.now(timezone.utc)
        self.session: Optional[AgentSession] = None
        # Performance metrics tracking
        self.metrics: dict = {
            "llm_first_chunk_times": [],
            "agent_turn_times": [],
            "interruption_count": 0,
            "user_utterances": 0,
        }
        self.last_user_speech_end: Optional[float] = None
        self.current_agent_turn_start: Optional[float] = None
        # Streaming debug logger for detailed pipeline analysis
        self.stream_logger: Optional[StreamingDebugLogger] = None
        # Production telephony optimizations
        self.network_monitor: Optional[NetworkMonitor] = None
        self.redis_manager: Optional[any] = None
        # Call metadata for JSON storage
        self.call_metadata: dict = {
            "call_id": call_id,
            "agent_id": agent_id,
            "started_at": self.started_at,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _load_agent_cached(agent_id: str) -> Optional[dict]:
    if agent_id in _AGENT_CACHE:
        return _AGENT_CACHE[agent_id]
    doc = await agents_repo.get_agent(agent_id)
    if doc:
        _AGENT_CACHE[agent_id] = doc
    return doc


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def entrypoint(ctx: JobContext) -> None:
    logger.info("Agent starting for room=%s", ctx.room.name)
    await ctx.connect()

    # ---- Parse room metadata ----
    raw_meta = ctx.room.metadata or "{}"
    try:
        meta = json.loads(raw_meta)
    except Exception:
        meta = {}
    agent_id = meta.get("agent_id")
    if agent_id is None:
        agent_id = "ag_0e66660d605c4fa7"
        logger.info("No agent_id in metadata, using default: %s", agent_id)
    
    call_id = meta.get("call_id")
    if call_id is None:
        call_id = f"call_{uuid.uuid4().hex[:16]}"
        logger.info("No call_id in metadata, generated: %s", call_id)

    if not agent_id or not call_id:
        logger.error("Missing agent_id/call_id in room metadata: %r", raw_meta)
        return

    # ---- Load agent config ----
    agent_doc = await _load_agent_cached(agent_id)
    if not agent_doc:
        logger.error("Unknown agent_id: %s", agent_id)
        return

    state = CallState(call_id=call_id, agent_id=agent_id)
    state.started_at = datetime.now(timezone.utc)
    
    # Initialize streaming debug logger for detailed pipeline analysis
    state.stream_logger = StreamingDebugLogger(call_id=call_id)
    state.stream_logger.log_event("INIT", f"Agent: {agent_id}, Room: {ctx.room.name}")
    
    # Initialize network monitor for production telephony
    state.network_monitor = NetworkMonitor(call_id=call_id, config=PRODUCTION_CONFIG.network_monitor)
    await state.network_monitor.start_monitoring()
    state.stream_logger.log_event("NETWORK", "Network quality monitoring started")
    
    # Get Redis manager for distributed state
    state.redis_manager = await get_redis_manager()
    if state.redis_manager._enabled:
        await state.redis_manager.save_call_state(call_id, {
            "agent_id": agent_id,
            "started_at": state.started_at.isoformat(),
            "status": "active",
        })
        state.stream_logger.log_event("REDIS", "Call state cached in Redis")

    # ---- Build pipeline with optimized settings ----
    vad = ctx.proc.userdata["vad"]
    
    # Deepgram STT — ultra-aggressive endpointing for low transcript_delay
    stt_config = PRODUCTION_CONFIG.stt
    stt = deepgram.STT(
        model=stt_config.model,
        language=agent_doc.get("language", "en"),
        interim_results=True,
        endpointing_ms=25,                            # was 100; rely on turn_detector for actual endpointing
        smart_format=False,
        filler_words=False,
        punctuate=False,
        no_delay=True,                                # send results ASAP, no batching
    )
    
    # Multi-provider LLM with ultra-low TTFT support
    llm_config = PRODUCTION_CONFIG.llm
    llm = create_llm(llm_config, agent_doc)
    
    # Log provider info
    provider_info = get_provider_info(llm_config)
    logger.info(f"LLM Provider: {provider_info['name']} ({provider_info['model']})")
    logger.info(f"Expected TTFT: {provider_info['expected_ttft_ms']}ms - {provider_info['description']}")
    
    if state.stream_logger:
        state.stream_logger.log_event("LLM", f"Provider: {provider_info['name']}, Model: {provider_info['model']}")
        state.stream_logger.log_event("LLM", f"Target TTFT: {provider_info['expected_ttft_ms']}ms")
    
    # Optimized Cartesia TTS with production telephony config
    tts_config = PRODUCTION_CONFIG.tts
    tts = cartesia.TTS(
        model=tts_config.model,
        language=agent_doc.get("language", "en"),
        voice=tts_config.voice,
        # Telephony-optimized audio format
        sample_rate=tts_config.sample_rate

    )
    
    stt.prewarm()
    
    logger.info("Pipeline built with optimized settings: STT endpointing=250ms, LLM temp=0.3")

    # Aggressive session configuration for sub-second e2e latency
    turn_detector = ctx.proc.userdata.get("turn_detector")
    session_kwargs = dict(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        preemptive_generation=True,     # generate response while user is still talking
        allow_interruptions=True,
        min_interruption_duration=0.05, # 50ms barge-in
        min_interruption_words=0,
        min_endpointing_delay=0.1,      # was 0.2 — trigger LLM faster on confident end-of-turn
        max_endpointing_delay=0.4,      # was 0.6 — cap waiting on turn_detector
    )
    if turn_detector is not None:
        session_kwargs["turn_detection"] = turn_detector
    session = AgentSession(**session_kwargs)

    # ── BIG visible startup banner so latency-relevant config is obvious in logs ──
    banner = (
        "\n" + "=" * 72 +
        f"\n  LATENCY CONFIG | call={call_id}"
        f"\n  STT  : Deepgram {stt_config.model}, endpointing=25ms, no_delay=True"
        f"\n  LLM  : {provider_info['name']} ({provider_info['model']}), max_tokens={llm_config.groq_max_tokens}"
        f"\n  TTS  : Cartesia {tts_config.model}, sample_rate={tts_config.sample_rate}"
        f"\n  TURN : {'turn_detector=ON' if turn_detector else 'turn_detector=OFF (VAD only)'}"
        f"\n  TIMING: min_endpoint=0.1s max_endpoint=0.4s preemptive=True interruption=50ms"
        f"\n  EXPECT: TTFT ≈ {provider_info['expected_ttft_ms']}ms + RTT, end_of_turn ≈ 300-500ms"
        + "\n" + "=" * 72
    )
    logger.info(banner)
    state.session = session

    dyn_agent = DynamicAgent(agent_doc=agent_doc, ctx=ctx, call_state=state)

    # ---- Comprehensive event logging (Cartesia-style) + Streaming Debug ----
    
    @session.on("user_speech_started")
    def _on_user_speech_started(ev) -> None:
        logger.info("-> 🧑🔊 User started speaking")
        if state.stream_logger:
            state.stream_logger.stt_user_speech_started()
    
    @session.on("user_speech_stopped")
    def _on_user_speech_stopped(ev) -> None:
        logger.info("-> 🧑🔇 User stopped speaking")
        if state.stream_logger:
            state.stream_logger.stt_user_speech_stopped()
    
    @session.on("agent_speech_started")
    def _on_agent_speech_started(ev) -> None:
        logger.info("-> 🤖🔊 Agent started speaking")
        state.current_agent_turn_start = time.time()
        if state.stream_logger:
            state.stream_logger.tts_playback_started()
    
    @session.on("agent_speech_stopped")
    def _on_agent_speech_stopped(ev) -> None:
        logger.info("-> 🤖🔇 Agent stopped speaking")
        
        # Calculate agent turn time
        if state.current_agent_turn_start:
            turn_time_ms = (time.time() - state.current_agent_turn_start) * 1000
            state.metrics["agent_turn_times"].append(turn_time_ms)
            logger.info(f"-> 📈 Log metric: agent_turn_ms={turn_time_ms:.2f}")
            state.current_agent_turn_start = None
        
        if state.stream_logger:
            state.stream_logger.tts_playback_stopped()
            state.stream_logger.log_turn_summary()
    
    @session.on("agent_speech_interrupted")
    def _on_agent_interrupted(ev) -> None:
        logger.info("-> 🤖❌ Agent interrupted")
        state.metrics["interruption_count"] += 1
        if state.stream_logger:
            state.stream_logger.tts_interrupted()
    
    @session.on("metrics_collected")
    def _on_metrics_collected(ev) -> None:
        # Slim handler — the verbose attribute dump was clogging the event loop.
        try:
            metrics_obj = getattr(ev, "metrics", None)
            if metrics_obj is None:
                return
            ttft = getattr(metrics_obj, "ttft", None)
            if ttft:
                first_chunk_ms = ttft * 1000
                state.metrics["llm_first_chunk_times"].append(first_chunk_ms)
                target = llm_config.target_ttft_ms
                if first_chunk_ms > target:
                    logger.warning("TTFT %.0fms exceeds target %dms", first_chunk_ms, target)
                else:
                    logger.info("TTFT %.0fms ✅", first_chunk_ms)
        except Exception:
            logger.exception("metrics_collected failed")

    # ---- Wire transcript collection ----
    @session.on("conversation_item_added")
    def _on_item_added(ev) -> None:
        try:
            item = ev.item
            role = getattr(item, "role", None)
            content = getattr(item, "text_content", None) or ""
            if role and content:
                msg = {
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                state.transcript.append(msg)
                
                # Log in Cartesia style
                emoji = "🧑" if role == "user" else "🤖"
                logger.info(f"-> {emoji}🗣️ {role.capitalize()} said: \"{content}\"")
                
                # Track LLM response and TTS synthesis for streaming analysis
                if state.stream_logger and role == "assistant":
                    # This is the complete LLM response - log it
                    state.stream_logger.llm_chunk_received(content)
                    state.stream_logger.llm_response_complete()
                    # TTS synthesis starts after LLM response
                    state.stream_logger.tts_synthesis_started(content)
        except Exception:
            logger.exception("on_item_added failed")

    # ---- Voicemail detection on user transcripts ----
    voicemail_enabled = agent_doc["tools"].get("voicemail_detection", {}).get("enabled")
    voicemail_message = agent_doc["tools"].get("voicemail_detection", {}).get(
        "voicemail_message", ""
    )

    @session.on("user_input_transcribed")
    def _on_user_transcript(ev) -> None:
        # Track user utterance metrics
        state.metrics["user_utterances"] += 1
        state.last_user_speech_end = time.time()
        
        # Get transcript text and check if it's interim or final
        text = getattr(ev, "transcript", "") or ""
        is_final = getattr(ev, "is_final", True)  # Default to final if not specified
        
        # Log to streaming debug logger
        if state.stream_logger:
            if is_final:
                state.stream_logger.stt_final_result(text)
                # Start LLM request tracking when we get final STT result
                state.stream_logger.llm_request_started(text)
            else:
                state.stream_logger.stt_interim_result(text)
        
        # Voicemail detection (only on final results)
        if not is_final:
            return
        if not voicemail_enabled or state.voicemail_detected:
            return
        elapsed = (datetime.now(timezone.utc) - state.started_at).total_seconds()
        if elapsed > VOICEMAIL_DETECT_WINDOW_SEC:
            return
        if VOICEMAIL_RE.search(text):
            state.voicemail_detected = True
            state.end_reason = "voicemail"
            state.tool_calls.append({
                "tool": "voicemail_detection",
                "args": {"matched": text},
                "at": datetime.now(timezone.utc).isoformat(),
            })
            logger.info("Voicemail detected: %r", text)
            asyncio.create_task(_handle_voicemail(session, ctx, voicemail_message))

    # ---- Start session ----
    await session.start(
        room=ctx.room,
        agent=dyn_agent,
        room_input_options=RoomInputOptions(close_on_disconnect=True),
    )

    # Capture SIP participant identity for transfer and store metadata
    state.call_metadata["room_name"] = ctx.room.name
    for p in ctx.room.remote_participants.values():
        if p.identity.startswith("phone-"):
            state.sip_participant_identity = p.identity
            # Extract phone number from identity (format: phone-1234567890)
            phone_num = p.identity.replace("phone-", "")
            if phone_num and not phone_num.startswith("+"):
                phone_num = "+" + phone_num
            state.call_metadata["phone_number"] = phone_num
            break

    # ---- Speak first message ----
    first_message = agent_doc.get("first_message") or "Hello! How can I help?"
    await session.say(first_message, allow_interruptions=True)

    # ---- Register shutdown hook (persist + webhook) ----
    async def _on_shutdown(reason: str = "") -> None:
        # Stop network monitoring and get final summary
        if state.network_monitor:
            await state.network_monitor.stop_monitoring()
            quality_summary = state.network_monitor.get_quality_summary()
            if state.stream_logger:
                state.stream_logger.log_event("NETWORK", f"Final quality: MOS={quality_summary['estimated_mos']}, Loss={quality_summary['packet_loss_rate']}%")
        
        # Update Redis with final state
        if state.redis_manager and state.redis_manager._enabled:
            await state.redis_manager.save_call_state(call_id, {
                "agent_id": agent_id,
                "status": "ended",
                "end_reason": reason,
                "ended_at": datetime.now(timezone.utc).isoformat(),
            })
            # Publish call ended event
            await state.redis_manager.publish_event("call_ended", {
                "call_id": call_id,
                "agent_id": agent_id,
                "reason": reason,
            })
        
        # Log call ended to streaming debug logger
        if state.stream_logger:
            state.stream_logger.log_call_ended()
        
        await _persist_and_webhook(state, agent_doc, reason)

    ctx.add_shutdown_callback(_on_shutdown)

    logger.info("Agent ready for call_id=%s agent_id=%s", call_id, agent_id)
    if state.stream_logger:
        state.stream_logger.log_event("READY", f"Agent ready, log file: {state.stream_logger.log_file}")


async def _handle_voicemail(
    session: AgentSession, ctx: JobContext, voicemail_message: str
) -> None:
    try:
        if voicemail_message:
            await session.say(voicemail_message, allow_interruptions=False)
        await asyncio.sleep(0.5)
        await ctx.delete_room()
    except Exception:
        logger.exception("voicemail handler failed")


async def _persist_and_webhook(
    state: CallState, agent_doc: dict, reason_hint: str
) -> None:
    end_reason = state.end_reason or (reason_hint or "hangup")
    
    # Calculate and log performance metrics summary
    metrics_summary = {
        "user_utterances": state.metrics["user_utterances"],
        "interruption_count": state.metrics["interruption_count"],
    }
    
    if state.metrics["llm_first_chunk_times"]:
        avg_first_chunk = sum(state.metrics["llm_first_chunk_times"]) / len(state.metrics["llm_first_chunk_times"])
        metrics_summary["avg_llm_first_chunk_ms"] = round(avg_first_chunk, 2)
        metrics_summary["min_llm_first_chunk_ms"] = round(min(state.metrics["llm_first_chunk_times"]), 2)
        metrics_summary["max_llm_first_chunk_ms"] = round(max(state.metrics["llm_first_chunk_times"]), 2)
    
    if state.metrics["agent_turn_times"]:
        avg_turn = sum(state.metrics["agent_turn_times"]) / len(state.metrics["agent_turn_times"])
        metrics_summary["avg_agent_turn_ms"] = round(avg_turn, 2)
        metrics_summary["min_agent_turn_ms"] = round(min(state.metrics["agent_turn_times"]), 2)
        metrics_summary["max_agent_turn_ms"] = round(max(state.metrics["agent_turn_times"]), 2)
    
    # Add network quality metrics if available
    if state.network_monitor:
        quality_summary = state.network_monitor.get_quality_summary()
        metrics_summary["network_quality"] = {
            "packet_loss_rate": quality_summary["packet_loss_rate"],
            "jitter_ms": quality_summary["jitter_ms"],
            "buffer_underruns": quality_summary["buffer_underruns"],
            "buffer_overruns": quality_summary["buffer_overruns"],
            "estimated_mos": quality_summary["estimated_mos"],
            "quality_rating": quality_summary["quality_rating"],
        }
    
    logger.info("📊 Call metrics summary: %s", metrics_summary)
    
    # Add metrics to call metadata
    state.call_metadata["metrics"] = metrics_summary
    
    try:
        await calls_repo.finalize_call(
            call_id=state.call_id,
            end_reason=end_reason,
            voicemail_detected=state.voicemail_detected,
            transcript=state.transcript,
            tool_calls=state.tool_calls,
            call_data=state.call_metadata,
        )
    except Exception:
        logger.exception("finalize_call failed")

    webhook_url = agent_doc.get("webhook_url")
    if not webhook_url:
        return

    payload = {
        "agent_id": state.agent_id,
        "call_id": state.call_id,
        "phone_number": None,  # filled below
        "started_at": state.started_at.isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "end_reason": end_reason,
        "voicemail_detected": state.voicemail_detected,
        "transcript": state.transcript,
        "tool_calls": state.tool_calls,
        "metrics": metrics_summary,  # Include performance metrics
    }
    # Pull phone_number from the persisted call record
    try:
        call_doc = await calls_repo.get_call(state.call_id)
        if call_doc:
            payload["phone_number"] = call_doc.get("phone_number")
            payload["room_name"] = call_doc.get("room_name")
            payload["sip_call_id"] = call_doc.get("sip_call_id")
            payload["duration_sec"] = call_doc.get("duration_sec")
    except Exception:
        pass

    code, err = await webhook.send_webhook(webhook_url, payload)
    try:
        await calls_repo.mark_webhook_sent(state.call_id, code, err)
    except Exception:
        logger.exception("mark_webhook_sent failed")
    logger.info("Webhook delivered: code=%s err=%s", code, err)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            num_idle_processes=2,
        )
    )
