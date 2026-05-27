import os
import logging
import sys
import asyncio
from pathlib import Path

from livekit.agents import (
    cli,
    WorkerOptions,
    JobContext,
    AgentSession,
    RunContext,
)
from livekit.agents.voice import Agent
from livekit.plugins import deepgram, openai, elevenlabs, silero, google

from dotenv import load_dotenv
load_dotenv()

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from RAGService import RAGService

# ------------------------------------------------------------
# Environment variables
# ------------------------------------------------------------
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

# ---------------- Logging ---------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("livekit-agent")

# ------------------------------------------------------------
# Agent Instructions
# ------------------------------------------------------------
AGENT_INSTRUCTIONS = """You are a helpful voice AI assistant. Be concise and conversational.

When relevant context is provided in your system context, use it naturally in your response. If no context is provided, answer based on your general knowledge."""

# ------------------------------------------------------------
# Custom Agent with Proactive RAG
# ------------------------------------------------------------
class MyAssistant(Agent):
    def __init__(self):
        # Initialize RAG service
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            
            self.rag_service = RAGService(
                openai_api_key=openai_api_key
            )
            logger.info("✓ RAG service initialized")
        except Exception as e:
            logger.error(f"RAG service initialization failed: {e}", exc_info=True)
            self.rag_service = None
        
        super().__init__(instructions=AGENT_INSTRUCTIONS)
    
    async def before_llm_inference(self, ctx: RunContext):
        """
        Hook that runs BEFORE the LLM is called. We inject RAG context here
        so the agent can speak immediately without tool call delays.
        """
        # Get the user's last message
        chat_ctx = ctx.chat_context
        if not chat_ctx or not chat_ctx.messages:
            return
        
        last_message = chat_ctx.messages[-1]
        if last_message.role != "user":
            return
        
        user_query = last_message.content
        logger.info(f"🔍 Proactive RAG search for: {user_query}")
        
        if not self.rag_service:
            return
        
        try:
            # Quick RAG search with timeout to prevent blocking
            search_results = await asyncio.wait_for(
                asyncio.to_thread(
                    self.rag_service.retrieval_based_search,
                    query=user_query,
                    collections=None,
                    top_k=1
                ),
                timeout=1.0  # 1 second max - adjust based on your needs
            )
            
            if search_results and len(search_results) > 0:
                context = search_results[0].get('text', '').strip()
                if context:
                    # Inject context into the chat as a system message
                    chat_ctx.append(
                        role="system",
                        text=f"Relevant context from knowledge base: {context}"
                    )
                    logger.info("✓ RAG context injected into chat")
            else:
                logger.info("No RAG results found")
                
        except asyncio.TimeoutError:
            logger.warning("⚠️ RAG search timed out - continuing without context")
        except Exception as e:
            logger.error(f"RAG search error: {e}")

# ---------------- Entrypoint ---------------- #
async def entrypoint(ctx: JobContext):
    logger.info("🚀 Agent starting...")

    # Connect to room
    await ctx.connect()
    logger.info("🔗 Connected to LiveKit room")

    # ---------------- STT - OPTIMIZED ---------------- #
    stt = deepgram.STT(
        model="nova-2-general",
        language="en",
        interim_results=True,
        endpointing_ms=200,
    )
    logger.info("✓ STT initialized (Deepgram - optimized)")

    # ---------------- LLM - OPTIMIZED ---------------- #
    llm = google.LLM(
        model="gemini-2.0-flash-exp",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,
    )
    logger.info("✓ LLM initialized (gemini-2.0-flash-exp)")

    # ---------------- TTS - OPTIMIZED ---------------- #
    tts = elevenlabs.TTS(
        base_url="https://api.eu.residency.elevenlabs.io/v1",
        voice_id="Xb7hH8MSUJpSbSDYk0k2",
        api_key=os.getenv("ELEVEN_API_KEY"),
        model="eleven_turbo_v2_5",
        language="en",
        streaming_latency=1,
        chunk_length_schedule=[80, 120, 150],
    )
    logger.info("✓ TTS initialized (ElevenLabs Turbo)")

    # ---------------- VAD - OPTIMIZED ---------------- #
    vad = silero.VAD.load(
        min_speech_duration=0.1,
        min_silence_duration=0.3,
    )
    logger.info("✓ VAD initialized (Silero - optimized)")

    # ---------------- Voice Agent ---------------- #
    assistant = MyAssistant()
    logger.info("🤖 Agent created with proactive RAG")

    # ---------------- Agent Session - OPTIMIZED ---------------- #
    session = AgentSession(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
    )
    logger.info("📋 Session created")

    # Start session
    await session.start(
        room=ctx.room,
        agent=assistant
    )

    logger.info("✅ Agent started successfully")

# ---------------- Worker ---------------- #
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )