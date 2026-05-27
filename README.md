# livekit_us_east-agent

Ultra-low-latency LiveKit voice agent for telephony, optimized for sub-second end-to-end latency when deployed in **us-east**.

**Stack:** LiveKit Agents · Deepgram (STT) · Groq (LLM) · Cartesia (TTS) · Silero VAD · LiveKit turn-detector · MongoDB (agent configs) · Redis (call state) · Twilio SIP

## Latency targets (when deployed in us-east-1)

| Stage | Target |
|---|---|
| `transcript_delay` | ~200 ms |
| `llm_ttft` | ~150 ms |
| `tts_ttfb` | ~120 ms |
| `end_of_turn` | 300–500 ms |
| **e2e** | **~700 ms** |

## Quick start

```bash
# 1. Install
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Fill in your API keys

# 3. Download model files (one-time)
python agent.py download-files

# 4. Run agent (dev mode)
python agent.py dev

# 5. Or talk to it directly from terminal
python agent.py console

# 6. Optional: run admin API
uvicorn app:app --reload --port 8000
```

## Why this is fast

- **`livekit-plugins-turn-detector`** (LLM-based end-of-utterance prediction) cuts end-of-turn from ~1.9s → ~0.4s.
- **Groq `llama-3.1-8b-instant`** for ~150 ms TTFT inside us-east.
- **Deepgram Nova-3** with `endpointing_ms=25` and `no_delay=True` for minimum transcript delay.
- **Cartesia Sonic-2** streaming TTS at 16 kHz.
- **`preemptive_generation=True`** — LLM starts while user is still finishing.
- **`min_interruption_duration=0.05`** — instant barge-in.
- System prompt auto-appends "reply in 1 short sentence" for telephony conciseness.

## Deployment

Run the agent in the **same region as your LLM/STT/TTS providers** (us-east-1 / `iad`). Network RTT from outside-region adds ~250 ms × 3 hops per turn.

| Region | Expected e2e |
|---|---|
| 🇮🇳 India laptop → US APIs | ~2000 ms |
| 🇺🇸 us-east-1 (AWS / Fly.io `iad`) | **~700 ms** |

## Files

| File | Purpose |
|---|---|
| `agent.py` | LiveKit worker entrypoint, pipeline wiring |
| `app.py` | FastAPI admin API for managing agents in MongoDB |
| `config.py` | `ProductionConfig` with VAD/STT/LLM/TTS tuning knobs |
| `llm_factory.py` | Multi-provider LLM factory (Groq / OpenAI / Gemini) |
| `streaming_logger.py` | Per-call structured streaming-debug logger |
| `network_monitor.py` | Packet loss / jitter / MOS estimation |
| `redis_manager.py` | Distributed call state + pub-sub |
| `services/` | Mongo repos (agents, calls, webhooks) |

## License

MIT
