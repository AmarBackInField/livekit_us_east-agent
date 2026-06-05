"""
LiveKit voice agent worker — static config (no MongoDB).

Uses hardcoded STATIC_AGENT_DOC for zero-latency agent config lookup.
All MongoDB calls removed for latency benchmarking.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

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
from livekit.plugins import cartesia, deepgram, silero
from livekit.plugins import mistralai as _mistralai_plugin  # noqa: F401 — must register on main thread

# Turn-detector — falls back gracefully if plugin not installed.
try:
    from livekit.plugins.turn_detector.english import EnglishModel as _TurnDetectorEN
    _TURN_DETECTOR_AVAILABLE = True
except Exception:
    _TurnDetectorEN = None  # type: ignore
    _TURN_DETECTOR_AVAILABLE = False

from streaming_logger import StreamingDebugLogger
from config import ProductionConfig, get_balanced_config
from network_monitor import NetworkMonitor
from redis_manager import get_redis_manager
from llm_factory import create_llm, get_provider_info

load_dotenv()

PRODUCTION_CONFIG = get_balanced_config()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("livekit-agent")

# ---------------------------------------------------------------------------
# Static agent config — replaces MongoDB lookup for latency testing
# ---------------------------------------------------------------------------

STATIC_AGENT_DOC: dict = {
    "system_prompt": (
        "You are Nova, an advanced AI assistant developed by CandexAI.\n\n"
        "Your role is to assist users with intelligent, accurate, and business-focused responses "
        "across enterprise workflows, AI automation, customer support, voice AI, document intelligence, "
        "and operational processes.\n\n"
        "About CandexAI:\n"
        "CandexAI builds enterprise-grade AI systems designed for organizations where privacy, security, "
        "control, and performance are essential. The platform provides AI agents, workflow automation, "
        "unified communication systems, and autonomous business process solutions.\n\n"
        "Your behavior guidelines:\n\n"
        "1. Be professional, concise, and intelligent.\n"
        "2. Prioritize clarity and actionable responses.\n"
        "3. Maintain a confident and enterprise-grade communication style.\n"
        "4. Support conversations related to:\n"
        "- AI Automation\n- Voice AI\n- Customer Support\n- Workflow Automation\n"
        "- CRM Integrations\n- AI Agents\n- Document Intelligence\n"
        "- Enterprise AI Solutions\n- Business Operations\n"
        "5. If users ask unrelated or harmful questions, politely redirect the conversation.\n"
        "6. Never generate misleading information or fake promises.\n"
        "7. Focus on solving business problems efficiently.\n"
        "8. Keep responses human-like and conversational.\n"
        "9. When appropriate, suggest AI-driven automation opportunities.\n"
        "10. Maintain data privacy and confidentiality in all interactions.\n\n"
        "Tone:\n- Smart\n- Modern\n- Helpful\n- Enterprise-focused\n- Efficient\n\n"
        "Identity:\n"
        "You are Nova by CandexAI — an enterprise AI assistant built to help businesses scale "
        "with intelligent automation."
    ),
    "first_message": "Hi Amar, how can I assist you?",
    "voice_id": "794f9389-aac1-45b6-b726-9d9369183238",
    "language": "en",
    "webhook_url": None,
    "tools": {
        "end_call":           {"enabled": True},
        "voicemail_detection": {"enabled": False, "voicemail_message": ""},
        "language_detection": {"enabled": False},
        "human_transfer":     {"enabled": False, "rules": []},
    },
}

# ---------------------------------------------------------------------------
# Voicemail patterns
# ---------------------------------------------------------------------------

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
VOICEMAIL_DETECT_WINDOW_SEC = 8.0


# ---------------------------------------------------------------------------
# Prewarm
# ---------------------------------------------------------------------------

_REGION_DIAG_DONE = False


def _run_region_diagnostic() -> None:
    """Measure TCP RTT to each provider from this container."""
    global _REGION_DIAG_DONE
    if _REGION_DIAG_DONE:
        return
    _REGION_DIAG_DONE = True

    import socket
    import time as _t
    from urllib.parse import urlparse

    targets = {
        "Groq":     "api.groq.com",
        "Deepgram": "api.deepgram.com",
        "Cartesia": "api.cartesia.ai",
        "Mistral":  "api.mistral.ai",
    }
    livekit_url = os.getenv("LIVEKIT_URL", "")
    if livekit_url:
        try:
            host = urlparse(livekit_url).hostname
            if host:
                targets["LiveKit"] = host
        except Exception:
            pass

    logger.info("─" * 72)
    logger.info("REGION DIAGNOSTIC (TCP connect RTT from this container)")
    for name, host in targets.items():
        try:
            t0 = _t.time()
            with socket.create_connection((host, 443), timeout=2.0):
                rtt_ms = (_t.time() - t0) * 1000
            warn = "  ⚠️ HIGH — wrong region?" if rtt_ms > 100 else ""
            logger.info(f"  {name:<10} → {host:<55} {rtt_ms:6.0f} ms{warn}")
        except Exception as e:
            logger.warning(f"  {name:<10} → {host:<55} FAIL ({e})")
    logger.info("─" * 72)


def prewarm(proc: JobProcess) -> None:
    """Preload Silero VAD + turn-detector once per worker process."""
    _run_region_diagnostic()
    proc.userdata["vad"] = silero.VAD.load(
        min_speech_duration=0.1,
        min_silence_duration=0.3,
        activation_threshold=0.6,
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
    logger.info("VAD prewarmed (min_speech=0.1s, min_silence=0.3s, threshold=0.6) - Balanced endpointing")


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class DynamicAgent(Agent):

    def __init__(self, *, agent_doc: dict, ctx: JobContext, call_state: "CallState"):
        self._agent_doc = agent_doc
        self._ctx = ctx
        self._state = call_state
        instructions = (
            agent_doc["system_prompt"].rstrip()
            + "\n\nIMPORTANT: This is a phone call. Reply in 1 short sentence "
            "(max 20 words). Be concise and conversational."
        )
        super().__init__(instructions=instructions)

    @function_tool()
    async def end_call(self) -> str:
        """End the current phone call. Use ONLY when the user clearly says
        goodbye, asks to hang up, or the conversation is naturally over."""
        if not self._agent_doc["tools"].get("end_call", {}).get("enabled"):
            return "end_call tool is not enabled for this agent."
        logger.info("Tool: end_call invoked")
        self._state.end_reason = "completed"
        self._state.tool_calls.append({"tool": "end_call", "at": datetime.now(timezone.utc).isoformat()})
        try:
            await self._state.session.say("Thank you for the call. Goodbye!", allow_interruptions=False)
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

    @function_tool()
    async def transfer_to_human(self, reason: str) -> str:
        """Transfer the caller to a human agent via SIP REFER."""
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
        self.metrics: dict = {
            "llm_first_chunk_times": [],
            "agent_turn_times": [],
            "interruption_count": 0,
            "user_utterances": 0,
        }
        self.last_user_speech_end: Optional[float] = None
        self.current_agent_turn_start: Optional[float] = None
        self.stream_logger: Optional[StreamingDebugLogger] = None
        self.network_monitor: Optional[NetworkMonitor] = None
        self.redis_manager: Optional[any] = None
        self.call_metadata: dict = {
            "call_id": call_id,
            "agent_id": agent_id,
            "started_at": self.started_at,
        }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def entrypoint(ctx: JobContext) -> None:
    logger.info("Agent starting for room=%s", ctx.room.name)
    await ctx.connect()

    # Parse room metadata for call_id only (agent_doc is now static)
    raw_meta = ctx.room.metadata or "{}"
    try:
        meta = json.loads(raw_meta)
    except Exception:
        meta = {}

    agent_id = meta.get("agent_id", "static_agent")
    call_id = meta.get("call_id") or f"call_{uuid.uuid4().hex[:16]}"
    agent_doc = STATIC_AGENT_DOC

    logger.info("Using static agent config | call_id=%s", call_id)

    state = CallState(call_id=call_id, agent_id=agent_id)
    state.started_at = datetime.now(timezone.utc)

    state.stream_logger = StreamingDebugLogger(call_id=call_id)
    state.stream_logger.log_event("INIT", f"Agent: {agent_id}, Room: {ctx.room.name}")

    state.network_monitor = NetworkMonitor(call_id=call_id, config=PRODUCTION_CONFIG.network_monitor)
    await state.network_monitor.start_monitoring()
    state.stream_logger.log_event("NETWORK", "Network quality monitoring started")

    state.redis_manager = await get_redis_manager()
    if state.redis_manager._enabled:
        await state.redis_manager.save_call_state(call_id, {
            "agent_id": agent_id,
            "started_at": state.started_at.isoformat(),
            "status": "active",
        })
        state.stream_logger.log_event("REDIS", "Call state cached in Redis")

    # ---- Build pipeline ----
    vad = ctx.proc.userdata["vad"]
    stt_config = PRODUCTION_CONFIG.stt
    llm_config = PRODUCTION_CONFIG.llm

    if llm_config.use_livekit_inference:
        stt_model = f"deepgram/{stt_config.model}:multi"
        logger.info("Using LiveKit Inference for STT: %s", stt_model)
    else:
        stt_model = stt_config.model

    stt = deepgram.STT(
        model=stt_model if not llm_config.use_livekit_inference else stt_config.model,
        language=agent_doc.get("language", "en"),
        endpointing_ms=250,
        smart_format=False,
        filler_words=False,
        punctuate=False,
        no_delay=True,
    )

    if llm_config.use_livekit_inference:
        llm_model = f"{llm_config.provider}/{llm_config.openai_model}"
        logger.info("Using LiveKit Inference for LLM: %s", llm_model)
    llm = create_llm(llm_config, agent_doc)

    provider_info = get_provider_info(llm_config)
    logger.info(f"LLM Provider: {provider_info['name']} ({provider_info['model']})")
    logger.info(f"Expected TTFT: {provider_info['expected_ttft_ms']}ms - {provider_info['description']}")

    if state.stream_logger:
        state.stream_logger.log_event("LLM", f"Provider: {provider_info['name']}, Model: {provider_info['model']}")
        state.stream_logger.log_event("LLM", f"Target TTFT: {provider_info['expected_ttft_ms']}ms")

    tts_config = PRODUCTION_CONFIG.tts
    # Use voice_id from static config
    voice_id = agent_doc.get("voice_id", tts_config.voice)

    if llm_config.use_livekit_inference:
        tts_voice = f"cartesia/{tts_config.model}:{voice_id}"
        logger.info("Using LiveKit Inference for TTS: %s", tts_voice)

    tts = cartesia.TTS(
        model=tts_config.model,
        language=agent_doc.get("language", "en"),
        voice=voice_id,
        sample_rate=tts_config.sample_rate,
    )

    stt.prewarm()

    inference_mode = "LiveKit Inference" if llm_config.use_livekit_inference else "Direct API"
    logger.info(f"Pipeline built: STT endpointing=250ms, Mode={inference_mode}")

    # ---- Session with turn handling ----
    _base_session_kwargs = dict(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        preemptive_generation=True,
        allow_interruptions=True,
        min_interruption_duration=0.05,
        min_interruption_words=0,
        min_endpointing_delay=0.15,
        max_endpointing_delay=0.5,
    )
    try:
        from livekit.agents import TurnHandlingOptions
        from livekit.plugins.turn_detector.multilingual import MultilingualModel
        multilingual_model = MultilingualModel()
        session_kwargs = {
            **_base_session_kwargs,
            "turn_handling": TurnHandlingOptions(turn_detection=multilingual_model),
        }
        logger.info("Using TurnHandlingOptions with MultilingualModel")
    except (ImportError, RuntimeError) as e:
        session_kwargs = _base_session_kwargs
        logger.warning(
            "MultilingualModel unavailable (%s); falling back to VAD-only endpointing. "
            "Run `python3 agent.py download-files` on the server to enable it.", e
        )
    session = AgentSession(**session_kwargs)

    banner = (
        "\n" + "=" * 72 +
        f"\n  LATENCY-OPTIMIZED CONFIG | call={call_id}"
        f"\n  MODE : LiveKit Inference {'✓ ENABLED' if llm_config.use_livekit_inference else '✗ DISABLED'}"
        f"\n  STT  : Deepgram {stt_config.model}, endpointing=250ms, no_delay=True"
        f"\n  LLM  : {provider_info['name']} ({provider_info['model']}), max_tokens={llm_config.mistral_max_tokens}"
        f"\n  TTS  : Cartesia {tts_config.model}, voice={voice_id}, sample_rate={tts_config.sample_rate}"
        f"\n  VAD  : min_speech=0.1s, min_silence=0.3s, threshold=0.6"
        f"\n  TIMING: min_endpoint=0.15s max_endpoint=0.5s preemptive=True interruption=50ms"
        f"\n  EXPECT: TTFT ≈ {provider_info['expected_ttft_ms']}ms, e2e latency < 1s"
        + "\n" + "=" * 72
    )
    logger.info(banner)
    state.session = session

    dyn_agent = DynamicAgent(agent_doc=agent_doc, ctx=ctx, call_state=state)

    # ---- Event handlers ----

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
        if state.current_agent_turn_start:
            turn_time_ms = (time.time() - state.current_agent_turn_start) * 1000
            state.metrics["agent_turn_times"].append(turn_time_ms)
            logger.info(f"-> 📈 agent_turn_ms={turn_time_ms:.2f}")
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
                if state.stream_logger:
                    state.stream_logger.llm_ttft_observed(first_chunk_ms)
        except Exception:
            logger.exception("metrics_collected failed")

    @session.on("conversation_item_added")
    def _on_item_added(ev) -> None:
        try:
            item = ev.item
            role = getattr(item, "role", None)
            content = getattr(item, "text_content", None) or ""
            if role and content:
                state.transcript.append({
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                emoji = "🧑" if role == "user" else "🤖"
                logger.info(f"-> {emoji}🗣️ {role.capitalize()} said: \"{content}\"")
                if state.stream_logger and role == "assistant":
                    state.stream_logger.llm_finalized(content)
                    state.stream_logger.tts_synthesis_started(content)
        except Exception:
            logger.exception("on_item_added failed")

    voicemail_enabled = agent_doc["tools"].get("voicemail_detection", {}).get("enabled")
    voicemail_message = agent_doc["tools"].get("voicemail_detection", {}).get("voicemail_message", "")

    @session.on("user_input_transcribed")
    def _on_user_transcript(ev) -> None:
        state.metrics["user_utterances"] += 1
        state.last_user_speech_end = time.time()
        text = getattr(ev, "transcript", "") or ""
        is_final = getattr(ev, "is_final", True)
        if state.stream_logger:
            if is_final:
                state.stream_logger.stt_final_result(text)
                state.stream_logger.llm_request_started(text)
            else:
                state.stream_logger.stt_interim_result(text)
        if not is_final or not voicemail_enabled or state.voicemail_detected:
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

    state.call_metadata["room_name"] = ctx.room.name
    for p in ctx.room.remote_participants.values():
        if p.identity.startswith("phone-"):
            state.sip_participant_identity = p.identity
            phone_num = p.identity.replace("phone-", "")
            if phone_num and not phone_num.startswith("+"):
                phone_num = "+" + phone_num
            state.call_metadata["phone_number"] = phone_num
            break

    await session.say(agent_doc["first_message"], allow_interruptions=True)

    async def _on_shutdown(reason: str = "") -> None:
        if state.network_monitor:
            await state.network_monitor.stop_monitoring()
            quality_summary = state.network_monitor.get_quality_summary()
            if state.stream_logger:
                state.stream_logger.log_event(
                    "NETWORK",
                    f"Final quality: MOS={quality_summary['estimated_mos']}, "
                    f"Loss={quality_summary['packet_loss_rate']}%"
                )
        if state.redis_manager and state.redis_manager._enabled:
            await state.redis_manager.save_call_state(call_id, {
                "agent_id": agent_id,
                "status": "ended",
                "end_reason": reason,
                "ended_at": datetime.now(timezone.utc).isoformat(),
            })
            await state.redis_manager.publish_event("call_ended", {
                "call_id": call_id,
                "agent_id": agent_id,
                "reason": reason,
            })
        if state.stream_logger:
            state.stream_logger.log_call_ended()
        _log_metrics_summary(state)

    ctx.add_shutdown_callback(_on_shutdown)
    logger.info("Agent ready for call_id=%s", call_id)
    if state.stream_logger:
        state.stream_logger.log_event("READY", f"Agent ready, log file: {state.stream_logger.log_file}")


async def _handle_voicemail(session: AgentSession, ctx: JobContext, voicemail_message: str) -> None:
    try:
        if voicemail_message:
            await session.say(voicemail_message, allow_interruptions=False)
        await asyncio.sleep(0.5)
        await ctx.delete_room()
    except Exception:
        logger.exception("voicemail handler failed")


def _log_metrics_summary(state: CallState) -> None:
    """Log call performance metrics to console (no DB write)."""
    m = state.metrics
    summary: dict = {
        "call_id": state.call_id,
        "user_utterances": m["user_utterances"],
        "interruption_count": m["interruption_count"],
    }
    if m["llm_first_chunk_times"]:
        summary["avg_ttft_ms"] = round(sum(m["llm_first_chunk_times"]) / len(m["llm_first_chunk_times"]), 2)
        summary["min_ttft_ms"] = round(min(m["llm_first_chunk_times"]), 2)
        summary["max_ttft_ms"] = round(max(m["llm_first_chunk_times"]), 2)
    if m["agent_turn_times"]:
        summary["avg_agent_turn_ms"] = round(sum(m["agent_turn_times"]) / len(m["agent_turn_times"]), 2)
        summary["min_agent_turn_ms"] = round(min(m["agent_turn_times"]), 2)
        summary["max_agent_turn_ms"] = round(max(m["agent_turn_times"]), 2)
    if state.network_monitor:
        qs = state.network_monitor.get_quality_summary()
        summary["network"] = {
            "mos": qs["estimated_mos"],
            "packet_loss": qs["packet_loss_rate"],
            "jitter_ms": qs["jitter_ms"],
            "quality": qs["quality_rating"],
        }
    logger.info("📊 Call metrics: %s", summary)


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
