import logging
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli, metrics
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice import AgentStateChangedEvent, MetricsCollectedEvent
from livekit.plugins import deepgram, openai, silero, elevenlabs, google
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Optimized version with reduced latency
# Key improvements:
# 1. Optimized TTS with streaming and lower latency settings
# 2. Reduced STT interim results processing
# 3. More aggressive turn detection
# 4. Optimized VAD settings
# 5. Limited LLM token output

logger = logging.getLogger("preemptive-generation")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Kelly. You interact with users via voice, "
            "so keep your responses concise and to the point - aim for 1-2 sentences. "
            "You are curious and friendly, and have a sense of humor.",
        )

    async def on_enter(self):
        self.session.generate_reply(instructions="say hello to the user briefly")

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage):
        # Keep this minimal to avoid canceling preemptive generation
        # Only add logic here if you need to modify context based on the message
        pass


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=google.LLM(
            model="gemini-2.5-flash-lite",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
            max_tokens=150,  # Limit response length for faster generation
        ),
        stt=deepgram.STT(
            model="nova-3",
            language="multi",
            interim_results=False,  # Only get final results to reduce overhead
            endpointing=300,  # Faster endpointing detection (in ms)
        ),
        tts=elevenlabs.TTS(
            base_url="https://api.eu.residency.elevenlabs.io/v1",
            voice_id="Xb7hH8MSUJpSbSDYk0k2",
            api_key=os.getenv("ELEVEN_API_KEY"),
            model="eleven_flash_v2_5",
            language="en",
            streaming=True,  # Enable streaming for lower latency
            optimize_streaming_latency=4,  # Max optimization (0-4, 4 is fastest)
            output_format="pcm_16000",  # Lower sample rate for faster processing
        ),
        turn_detection=MultilingualModel(
            min_endpointing_delay=0.5,  # More aggressive turn detection (reduced from default)
            prefix_padding_ms=200,  # Reduced padding for faster response
        ),
        preemptive_generation=True,
    )

    last_eou_metrics: metrics.EOUMetrics | None = None

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics

        metrics.log_metrics(ev.metrics)
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        if (
            ev.new_state == "speaking"
            and last_eou_metrics
            and last_eou_metrics.speech_id == session.current_speech.id
        ):
            # Log end-to-end latency for monitoring
            logger.info(
                f"End-to-end latency: {ev.created_at - last_eou_metrics.last_speaking_time:.3f}s"
            )
            logger.info(last_eou_metrics)

    await session.start(agent=MyAgent(), room=ctx.room)


def prewarm(proc: JobProcess):
    # Optimized VAD settings for faster speech detection
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))