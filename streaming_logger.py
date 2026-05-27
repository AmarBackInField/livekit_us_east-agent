"""
Streaming Debug Logger for Voice Pipeline Analysis.

Tracks and logs detailed timing information for:
- STT (Speech-to-Text) streaming chunks
- LLM streaming tokens/chunks
- TTS (Text-to-Speech) audio generation

Creates a separate log file per call for easy analysis.
"""

import logging
import os
import time
from datetime import datetime
from typing import Optional


class StreamingDebugLogger:
    """
    Dedicated logger for tracking streaming behavior of voice pipeline components.
    Creates a separate log file per call with detailed timing information.
    """

    def __init__(self, call_id: str, log_dir: str = "logs"):
        self.call_id = call_id
        self.log_dir = log_dir
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Create unique log file for this call
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = os.path.join(log_dir, f"streaming_debug_{timestamp}_{call_id}.log")
        
        # Set up file handler
        self.logger = logging.getLogger(f"streaming_debug_{call_id}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear any existing handlers
        
        # File handler with detailed format
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Also log to console for real-time monitoring
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Timing trackers
        self._stt_start_time: Optional[float] = None
        self._stt_interim_count: int = 0
        self._stt_last_interim_time: Optional[float] = None
        
        self._llm_start_time: Optional[float] = None
        self._llm_first_chunk_time: Optional[float] = None
        self._llm_chunk_count: int = 0
        self._llm_last_chunk_time: Optional[float] = None
        self._llm_full_response: str = ""
        
        self._tts_start_time: Optional[float] = None
        self._tts_first_audio_time: Optional[float] = None
        self._tts_chunk_count: int = 0
        
        self._user_speech_end_time: Optional[float] = None
        self._agent_speech_start_time: Optional[float] = None
        
        # Log header
        self.logger.info("=" * 80)
        self.logger.info(f"[INIT] 🎯 Streaming Debug Log Started for call_id: {call_id}")
        self.logger.info(f"[INIT] 📁 Log file: {self.log_file}")
        self.logger.info("=" * 80)
    
    # =========================================================================
    # STT (Speech-to-Text) Logging
    # =========================================================================
    
    def stt_user_speech_started(self):
        """Log when user starts speaking."""
        self._stt_start_time = time.time()
        self._stt_interim_count = 0
        self.logger.info("[STT] 🎤 User speech STARTED")
    
    def stt_user_speech_stopped(self):
        """Log when user stops speaking."""
        self._user_speech_end_time = time.time()
        if self._stt_start_time:
            duration = (self._user_speech_end_time - self._stt_start_time) * 1000
            self.logger.info(f"[STT] 🔇 User speech STOPPED (duration: {duration:.0f}ms)")
        else:
            self.logger.info("[STT] 🔇 User speech STOPPED")
    
    def stt_interim_result(self, text: str):
        """Log interim (partial) STT result - indicates streaming is working."""
        now = time.time()
        self._stt_interim_count += 1
        
        if self._stt_last_interim_time:
            delta = (now - self._stt_last_interim_time) * 1000
            self.logger.info(f"[STT] 📝 Interim #{self._stt_interim_count}: \"{text}\" (+{delta:.0f}ms)")
        else:
            if self._stt_start_time:
                delta = (now - self._stt_start_time) * 1000
                self.logger.info(f"[STT] 📝 Interim #{self._stt_interim_count}: \"{text}\" (first at {delta:.0f}ms)")
            else:
                self.logger.info(f"[STT] 📝 Interim #{self._stt_interim_count}: \"{text}\"")
        
        self._stt_last_interim_time = now
    
    def stt_final_result(self, text: str):
        """Log final STT result."""
        now = time.time()
        self._user_speech_end_time = now
        
        if self._stt_start_time:
            total_time = (now - self._stt_start_time) * 1000
            self.logger.info(f"[STT] ✅ FINAL: \"{text}\"")
            self.logger.info(f"[STT] ⏱️  Total STT time: {total_time:.0f}ms, Interim chunks: {self._stt_interim_count}")
            
            if self._stt_interim_count == 0:
                self.logger.warning("[STT] ⚠️  NO INTERIM RESULTS - STT may not be streaming!")
            else:
                self.logger.info(f"[STT] ✅ STT IS STREAMING ({self._stt_interim_count} interim results)")
        else:
            self.logger.info(f"[STT] ✅ FINAL: \"{text}\"")
        
        # Reset for next utterance
        self._stt_last_interim_time = None
    
    # =========================================================================
    # LLM Logging
    # =========================================================================
    
    def llm_request_started(self, prompt_preview: str = ""):
        """Log when LLM request starts."""
        self._llm_start_time = time.time()
        self._llm_first_chunk_time = None
        self._llm_chunk_count = 0
        self._llm_full_response = ""
        
        self.logger.info("[LLM] 🚀 Request STARTED")
        if prompt_preview:
            preview = prompt_preview[:100] + "..." if len(prompt_preview) > 100 else prompt_preview
            self.logger.info(f"[LLM] 📤 Prompt preview: \"{preview}\"")
    
    def llm_chunk_received(self, chunk_text: str):
        """Log LLM response committed to conversation."""
        now = time.time()
        self._llm_chunk_count += 1
        self._llm_full_response += chunk_text
        
        # Truncate long responses for logging
        display_text = chunk_text[:80] + "..." if len(chunk_text) > 80 else chunk_text
        display_text = display_text.replace("\n", " ")
        
        if self._llm_first_chunk_time is None:
            # First response committed - this is the critical TTFC metric
            self._llm_first_chunk_time = now
            if self._llm_start_time:
                ttfc = (now - self._llm_start_time) * 1000
                self.logger.info(f"[LLM] 📦 Response #{self._llm_chunk_count}: \"{display_text}\" (⚡ COMMITTED at {ttfc:.0f}ms)")
                
                # Analyze the TTFC
                if ttfc > 3000:
                    self.logger.warning(f"[LLM] 🐌 VERY SLOW TTFC: {ttfc:.0f}ms - LLM is likely NOT streaming to TTS!")
                elif ttfc > 1500:
                    self.logger.warning(f"[LLM] ⚠️  SLOW TTFC: {ttfc:.0f}ms - Consider optimizing LLM")
                else:
                    self.logger.info(f"[LLM] ✅ Good TTFC: {ttfc:.0f}ms")
            else:
                self.logger.info(f"[LLM] 📦 Response #{self._llm_chunk_count}: \"{display_text}\" (FIRST)")
        else:
            if self._llm_last_chunk_time:
                delta = (now - self._llm_last_chunk_time) * 1000
                self.logger.info(f"[LLM] 📦 Response #{self._llm_chunk_count}: \"{display_text}\" (+{delta:.0f}ms)")
            else:
                self.logger.info(f"[LLM] 📦 Response #{self._llm_chunk_count}: \"{display_text}\"")
        
        self._llm_last_chunk_time = now
    
    def llm_response_complete(self):
        """Log when LLM response is complete."""
        now = time.time()
        
        if self._llm_start_time:
            total_time = (now - self._llm_start_time) * 1000
            ttfc = (self._llm_first_chunk_time - self._llm_start_time) * 1000 if self._llm_first_chunk_time else 0
            
            # Truncate full response for logging
            full_display = self._llm_full_response[:100] + "..." if len(self._llm_full_response) > 100 else self._llm_full_response
            full_display = full_display.replace("\n", " ")
            
            self.logger.info(f"[LLM] ✅ COMPLETE: \"{full_display}\"")
            self.logger.info(f"[LLM] ⏱️  Total: {total_time:.0f}ms | TTFC: {ttfc:.0f}ms | Responses: {self._llm_chunk_count}")
            
            # Analysis
            if self._llm_chunk_count <= 1 and ttfc > 2000:
                self.logger.warning("[LLM] 🚨 CRITICAL: Single response after long delay = NO STREAMING!")
                self.logger.warning("[LLM] 💡 The LLM is generating the FULL response before sending to TTS")
                self.logger.warning("[LLM] 💡 This is the main latency bottleneck!")
            elif self._llm_chunk_count <= 1:
                self.logger.info("[LLM] ℹ️  Single response (short response or buffered)")
            else:
                self.logger.info(f"[LLM] ✅ Multiple responses committed ({self._llm_chunk_count})")
        else:
            full_display = self._llm_full_response[:100] + "..." if len(self._llm_full_response) > 100 else self._llm_full_response
            self.logger.info(f"[LLM] ✅ COMPLETE: \"{full_display}\"")
        
        # Reset for next response
        self._llm_last_chunk_time = None
    
    # =========================================================================
    # TTS (Text-to-Speech) Logging
    # =========================================================================
    
    def tts_synthesis_started(self, text: str = ""):
        """Log when TTS synthesis starts."""
        self._tts_start_time = time.time()
        self._tts_first_audio_time = None
        self._tts_chunk_count = 0
        
        self.logger.info(f"[TTS] 🔊 Synthesis STARTED")
        if text:
            preview = text[:50] + "..." if len(text) > 50 else text
            self.logger.info(f"[TTS] 📝 Text: \"{preview}\"")
    
    def tts_audio_chunk_generated(self):
        """Log each TTS audio chunk - indicates streaming is working."""
        now = time.time()
        self._tts_chunk_count += 1
        
        if self._tts_first_audio_time is None:
            self._tts_first_audio_time = now
            if self._tts_start_time:
                ttfa = (now - self._tts_start_time) * 1000
                self.logger.info(f"[TTS] 🎵 Audio chunk #{self._tts_chunk_count} (⚡ FIRST AUDIO at {ttfa:.0f}ms)")
            else:
                self.logger.info(f"[TTS] 🎵 Audio chunk #{self._tts_chunk_count} (FIRST AUDIO)")
        else:
            self.logger.debug(f"[TTS] 🎵 Audio chunk #{self._tts_chunk_count}")
    
    def tts_playback_started(self):
        """Log when audio playback actually starts."""
        self._agent_speech_start_time = time.time()
        
        if self._tts_start_time:
            delay = (self._agent_speech_start_time - self._tts_start_time) * 1000
            self.logger.info(f"[TTS] 🔈 Playback STARTED (delay from synthesis start: {delay:.0f}ms)")
        else:
            self.logger.info("[TTS] 🔈 Playback STARTED")
        
        # Calculate end-to-end latency
        if self._user_speech_end_time:
            e2e = (self._agent_speech_start_time - self._user_speech_end_time) * 1000
            self.logger.info(f"[PIPELINE] ⚡ End-to-end latency: {e2e:.0f}ms (user speech end → agent speech start)")
    
    def tts_playback_stopped(self):
        """Log when audio playback stops."""
        now = time.time()
        
        if self._tts_start_time:
            total_time = (now - self._tts_start_time) * 1000
            self.logger.info(f"[TTS] ✅ Playback STOPPED (total TTS time: {total_time:.0f}ms, chunks: {self._tts_chunk_count})")
            
            if self._tts_chunk_count <= 1:
                self.logger.warning("[TTS] ⚠️  ONLY 1 CHUNK - TTS may not be streaming!")
            else:
                self.logger.info(f"[TTS] ✅ TTS IS STREAMING ({self._tts_chunk_count} audio chunks)")
        else:
            self.logger.info("[TTS] ✅ Playback STOPPED")
    
    def tts_interrupted(self):
        """Log when TTS is interrupted by user."""
        now = time.time()
        self.logger.info("[TTS] ❌ INTERRUPTED by user")
        
        if self._agent_speech_start_time:
            speaking_time = (now - self._agent_speech_start_time) * 1000
            self.logger.info(f"[TTS] ⏱️  Agent was speaking for: {speaking_time:.0f}ms before interruption")
    
    # =========================================================================
    # Pipeline Summary
    # =========================================================================
    
    def log_turn_summary(self):
        """Log a summary of the complete turn."""
        self.logger.info("-" * 60)
        self.logger.info("[SUMMARY] 📊 Turn Summary:")
        
        # STT summary
        if self._stt_interim_count > 0:
            self.logger.info(f"[SUMMARY] STT: ✅ Streaming ({self._stt_interim_count} interim results)")
        else:
            self.logger.info("[SUMMARY] STT: ⚠️  Not streaming (0 interim results)")
        
        # LLM summary
        if self._llm_chunk_count > 1:
            ttfc = (self._llm_first_chunk_time - self._llm_start_time) * 1000 if self._llm_first_chunk_time and self._llm_start_time else 0
            self.logger.info(f"[SUMMARY] LLM: ✅ Streaming ({self._llm_chunk_count} chunks, TTFC: {ttfc:.0f}ms)")
        elif self._llm_chunk_count == 1:
            self.logger.info("[SUMMARY] LLM: ⚠️  Not streaming (only 1 chunk)")
        else:
            self.logger.info("[SUMMARY] LLM: ❓ No data")
        
        # TTS summary
        if self._tts_chunk_count > 1:
            self.logger.info(f"[SUMMARY] TTS: ✅ Streaming ({self._tts_chunk_count} audio chunks)")
        elif self._tts_chunk_count == 1:
            self.logger.info("[SUMMARY] TTS: ⚠️  Not streaming (only 1 chunk)")
        else:
            self.logger.info("[SUMMARY] TTS: ❓ No data")
        
        # E2E latency
        if self._user_speech_end_time and self._agent_speech_start_time:
            e2e = (self._agent_speech_start_time - self._user_speech_end_time) * 1000
            self.logger.info(f"[SUMMARY] E2E Latency: {e2e:.0f}ms")
        
        self.logger.info("-" * 60)
    
    def log_call_ended(self):
        """Log when the call ends."""
        self.logger.info("=" * 80)
        self.logger.info("[END] 📞 Call ended")
        self.logger.info(f"[END] 📁 Full log saved to: {self.log_file}")
        self.logger.info("=" * 80)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def log_event(self, component: str, message: str):
        """Log a generic event."""
        self.logger.info(f"[{component}] {message}")
    
    def log_warning(self, component: str, message: str):
        """Log a warning."""
        self.logger.warning(f"[{component}] ⚠️  {message}")
    
    def log_error(self, component: str, message: str):
        """Log an error."""
        self.logger.error(f"[{component}] ❌ {message}")
