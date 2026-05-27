"""
Production Telephony Configuration

Centralized configuration for buffer management, jitter handling,
and chunk size optimization for SIP/telephony deployments.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioBufferConfig:
    """Audio buffer configuration for telephony stability."""
    
    # Buffer sizes (in milliseconds)
    min_buffer_ms: int = 20  # Minimum buffer for low latency
    target_buffer_ms: int = 40  # Target buffer for normal operation
    max_buffer_ms: int = 80  # Maximum buffer for high jitter networks
    
    # Jitter buffer settings
    enable_adaptive_jitter: bool = True
    jitter_buffer_min_ms: int = 20
    jitter_buffer_max_ms: int = 60
    
    # Packet loss tolerance
    max_packet_loss_percent: float = 3.0  # Alert if exceeds
    enable_packet_loss_concealment: bool = True


@dataclass
class STTConfig:
    """Speech-to-Text configuration optimized for telephony."""
    
    model: str = "nova-3"
    sample_rate: int = 16000  # Optimal for telephony (8000 or 16000)
    encoding: str = "linear16"
    interim_results: bool = True
    endpointing_ms: int = 100   # Aggressive: rely on turn_detector for true endpointing
    smart_format: bool = False
    filler_words: bool = False
    
    # Telephony-specific
    enable_automatic_punctuation: bool = False  # Reduce processing
    profanity_filter: bool = False  # Reduce latency
    
    # Buffer settings
    chunk_size_ms: int = 20  # Audio chunk size for streaming


@dataclass
class LLMConfig:
    """LLM configuration supporting multiple providers for ultra-low TTFT."""
    
    # Provider selection: "openai", "google", "groq"
    provider: str = "groq"  # Default to Groq for fastest TTFT (100-200ms)
    
    # OpenAI settings
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.3
    openai_max_tokens: int = 80
    
    # Google Gemini settings
    google_model: str = "gemini-2.5-flash"
    google_temperature: float = 0.3
    google_max_tokens: int = 80
    
    # Groq settings (fastest) — 8B-instant for sub-200ms TTFT
    groq_model: str = "llama-3.1-8b-instant"
    groq_temperature: float = 0.3
    groq_max_tokens: int = 80    # Short answers = fast responses on phone
    
    # Performance targets
    target_ttft_ms: int = 200  # Alert if exceeded
    
    # Streaming optimization
    stream: bool = True
    timeout_seconds: int = 30
    
    # Retry configuration
    max_retries: int = 2
    retry_delay_ms: int = 100
    
    # Fallback configuration
    enable_fallback: bool = True
    fallback_provider: str = "openai"  # Fallback if primary fails


@dataclass
class TTSConfig:
    """Text-to-Speech configuration optimized for telephony."""
    
    model: str = "sonic-2"
    voice: str = "f786b574-daa5-4673-aa0c-cbe3e8534c02"
    
    # Audio format for telephony
    sample_rate: int = 16000  # 8000 for narrow-band, 16000 for wide-band
    encoding: str = "pcm_16000"
    
    # Chunk settings for smooth playback
    chunk_length_ms: int = 50   # Smaller chunks = faster first audio frame
    enable_word_timestamps: bool = False  # Reduce overhead


@dataclass
class NetworkMonitorConfig:
    """Network quality monitoring configuration."""
    
    # Monitoring intervals
    packet_loss_check_interval_ms: int = 1000
    jitter_measurement_window_ms: int = 5000
    
    # Thresholds for alerts
    packet_loss_warning_threshold: float = 1.0  # 1%
    packet_loss_critical_threshold: float = 3.0  # 3%
    jitter_warning_threshold_ms: float = 30.0
    jitter_critical_threshold_ms: float = 50.0
    
    # Buffer monitoring
    buffer_underrun_threshold: int = 5  # Alert after N underruns
    buffer_overrun_threshold: int = 10  # Alert after N overruns
    
    # Quality metrics
    calculate_mos_score: bool = True  # Mean Opinion Score estimation
    log_quality_metrics: bool = True


@dataclass
class RedisConfig:
    """Redis configuration for distributed state management."""
    
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    
    # Connection pool settings
    max_connections: int = 500  # For 100-500 concurrent calls
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    
    # Key TTLs (in seconds)
    call_state_ttl: int = 3600  # 1 hour
    session_cache_ttl: int = 300  # 5 minutes
    metrics_ttl: int = 86400  # 24 hours
    
    # Pub/sub channels
    events_channel: str = "agent:events"
    metrics_channel: str = "agent:metrics"
    alerts_channel: str = "agent:alerts"


@dataclass
class SIPConfig:
    """SIP-specific configuration."""
    
    # Connection settings
    enable_persistent_connections: bool = True
    connection_timeout_seconds: int = 30
    
    # RTP settings
    rtp_packet_size_ms: int = 20  # Standard for telephony
    enable_dtmf_detection: bool = True
    
    # Codec preferences (in order)
    preferred_codecs: list = None
    
    def __post_init__(self):
        if self.preferred_codecs is None:
            self.preferred_codecs = ["opus", "pcmu", "pcma"]  # Opus best, G.711 fallback


@dataclass
class ProductionConfig:
    """Master configuration for production telephony deployment."""
    
    # Component configs
    audio_buffer: AudioBufferConfig = None
    stt: STTConfig = None
    llm: LLMConfig = None
    tts: TTSConfig = None
    network_monitor: NetworkMonitorConfig = None
    redis: RedisConfig = None
    sip: SIPConfig = None
    
    # Global settings
    enable_detailed_logging: bool = True
    log_network_metrics: bool = True
    enable_auto_recovery: bool = True
    
    # Concurrency limits
    max_concurrent_calls: int = 500
    worker_processes: int = 4
    calls_per_worker: int = 125
    
    def __post_init__(self):
        # Initialize sub-configs with defaults
        if self.audio_buffer is None:
            self.audio_buffer = AudioBufferConfig()
        if self.stt is None:
            self.stt = STTConfig()
        if self.llm is None:
            self.llm = LLMConfig()
        if self.tts is None:
            self.tts = TTSConfig()
        if self.network_monitor is None:
            self.network_monitor = NetworkMonitorConfig()
        if self.redis is None:
            self.redis = RedisConfig()
        if self.sip is None:
            self.sip = SIPConfig()


# Default production configuration
DEFAULT_PRODUCTION_CONFIG = ProductionConfig()


# Configuration presets for different scenarios

def get_low_latency_config() -> ProductionConfig:
    """Configuration optimized for lowest latency (may sacrifice stability)."""
    config = ProductionConfig()
    config.audio_buffer.target_buffer_ms = 20
    config.audio_buffer.max_buffer_ms = 40
    config.stt.endpointing_ms = 200
    config.tts.chunk_length_ms = 50
    return config


def get_high_stability_config() -> ProductionConfig:
    """Configuration optimized for stability on poor networks."""
    config = ProductionConfig()
    config.audio_buffer.target_buffer_ms = 60
    config.audio_buffer.max_buffer_ms = 100
    config.audio_buffer.max_packet_loss_percent = 5.0
    config.stt.endpointing_ms = 300
    config.tts.chunk_length_ms = 150
    return config


def get_balanced_config() -> ProductionConfig:
    """Balanced configuration (default)."""
    return DEFAULT_PRODUCTION_CONFIG
