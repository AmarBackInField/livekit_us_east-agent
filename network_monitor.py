"""
Network Quality Monitor

Tracks packet loss, jitter, buffer health, and audio quality metrics
for production telephony deployments.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Deque
import logging

from config import NetworkMonitorConfig

logger = logging.getLogger("network-monitor")


@dataclass
class PacketStats:
    """Statistics for packet transmission."""
    sent: int = 0
    received: int = 0
    lost: int = 0
    
    @property
    def loss_rate(self) -> float:
        """Calculate packet loss rate as percentage."""
        if self.sent == 0:
            return 0.0
        return (self.lost / self.sent) * 100.0


@dataclass
class JitterStats:
    """Jitter measurement statistics."""
    samples: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    
    def add_sample(self, jitter_ms: float):
        """Add a jitter measurement sample."""
        self.samples.append(jitter_ms)
    
    @property
    def current_jitter_ms(self) -> float:
        """Get current average jitter."""
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)
    
    @property
    def max_jitter_ms(self) -> float:
        """Get maximum jitter in current window."""
        if not self.samples:
            return 0.0
        return max(self.samples)


@dataclass
class BufferStats:
    """Buffer health statistics."""
    underruns: int = 0
    overruns: int = 0
    current_depth_ms: float = 0.0
    target_depth_ms: float = 40.0
    
    @property
    def health_score(self) -> float:
        """Calculate buffer health score (0-100)."""
        # Penalize underruns/overruns
        penalty = (self.underruns + self.overruns) * 5
        score = 100 - penalty
        
        # Penalize deviation from target depth
        depth_deviation = abs(self.current_depth_ms - self.target_depth_ms)
        score -= depth_deviation / 2
        
        return max(0.0, min(100.0, score))


@dataclass
class AudioQualityMetrics:
    """Audio quality metrics."""
    estimated_mos: float = 0.0  # Mean Opinion Score (1-5)
    signal_level_db: float = 0.0
    noise_level_db: float = 0.0
    
    @property
    def snr_db(self) -> float:
        """Signal-to-Noise Ratio."""
        return self.signal_level_db - self.noise_level_db


class NetworkMonitor:
    """
    Monitor network quality and audio metrics for telephony calls.
    
    Tracks:
    - Packet loss rate
    - Jitter (timing variation)
    - Buffer underruns/overruns
    - Audio quality (MOS score estimation)
    - Round-trip time
    """
    
    def __init__(self, call_id: str, config: Optional[NetworkMonitorConfig] = None):
        self.call_id = call_id
        self.config = config or NetworkMonitorConfig()
        
        # Statistics
        self.packet_stats = PacketStats()
        self.jitter_stats = JitterStats()
        self.buffer_stats = BufferStats()
        self.audio_quality = AudioQualityMetrics()
        
        # Timing
        self.start_time = time.time()
        self.last_packet_time: Optional[float] = None
        self.rtt_samples: Deque[float] = deque(maxlen=50)
        
        # Monitoring state
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        logger.info(f"[{call_id}] Network monitor initialized")
    
    def record_packet_sent(self):
        """Record a packet being sent."""
        self.packet_stats.sent += 1
    
    def record_packet_received(self, arrival_time: Optional[float] = None):
        """Record a packet being received."""
        self.packet_stats.received += 1
        
        # Calculate jitter if we have timing info
        if arrival_time and self.last_packet_time:
            # Jitter is variation in packet arrival time
            inter_arrival = arrival_time - self.last_packet_time
            if hasattr(self, '_last_inter_arrival'):
                jitter = abs(inter_arrival - self._last_inter_arrival) * 1000  # Convert to ms
                self.jitter_stats.add_sample(jitter)
            self._last_inter_arrival = inter_arrival
        
        self.last_packet_time = arrival_time or time.time()
    
    def record_packet_lost(self):
        """Record a packet loss."""
        self.packet_stats.lost += 1
        
        # Log if exceeds threshold
        loss_rate = self.packet_stats.loss_rate
        if loss_rate > self.config.packet_loss_critical_threshold:
            logger.error(f"[{self.call_id}] CRITICAL packet loss: {loss_rate:.2f}%")
        elif loss_rate > self.config.packet_loss_warning_threshold:
            logger.warning(f"[{self.call_id}] High packet loss: {loss_rate:.2f}%")
    
    def record_buffer_underrun(self):
        """Record a buffer underrun (buffer empty, audio dropout)."""
        self.buffer_stats.underruns += 1
        logger.warning(f"[{self.call_id}] Buffer underrun #{self.buffer_stats.underruns}")
        
        if self.buffer_stats.underruns >= self.config.buffer_underrun_threshold:
            logger.error(f"[{self.call_id}] Excessive buffer underruns: {self.buffer_stats.underruns}")
    
    def record_buffer_overrun(self):
        """Record a buffer overrun (buffer full, packets dropped)."""
        self.buffer_stats.overruns += 1
        logger.warning(f"[{self.call_id}] Buffer overrun #{self.buffer_stats.overruns}")
        
        if self.buffer_stats.overruns >= self.config.buffer_overrun_threshold:
            logger.error(f"[{self.call_id}] Excessive buffer overruns: {self.buffer_stats.overruns}")
    
    def update_buffer_depth(self, depth_ms: float):
        """Update current buffer depth."""
        self.buffer_stats.current_depth_ms = depth_ms
    
    def record_rtt(self, rtt_ms: float):
        """Record round-trip time measurement."""
        self.rtt_samples.append(rtt_ms)
    
    def estimate_mos_score(self) -> float:
        """
        Estimate Mean Opinion Score (MOS) for audio quality.
        
        MOS scale: 1 (bad) to 5 (excellent)
        Based on E-Model (ITU-T G.107)
        """
        # Start with perfect score
        r_factor = 93.2
        
        # Degrade based on packet loss
        loss_rate = self.packet_stats.loss_rate
        if loss_rate > 0:
            # Packet loss has exponential impact
            r_factor -= (loss_rate * 2.5) + (loss_rate ** 2 * 0.1)
        
        # Degrade based on jitter
        jitter = self.jitter_stats.current_jitter_ms
        if jitter > 0:
            # Jitter impact (normalized)
            r_factor -= jitter / 10.0
        
        # Degrade based on latency (RTT)
        if self.rtt_samples:
            avg_rtt = sum(self.rtt_samples) / len(self.rtt_samples)
            if avg_rtt > 150:  # Noticeable above 150ms
                r_factor -= (avg_rtt - 150) / 40.0
        
        # Convert R-factor to MOS (ITU-T G.107)
        if r_factor < 0:
            mos = 1.0
        elif r_factor > 100:
            mos = 4.5
        else:
            mos = 1 + 0.035 * r_factor + 7e-6 * r_factor * (r_factor - 60) * (100 - r_factor)
        
        self.audio_quality.estimated_mos = max(1.0, min(5.0, mos))
        return self.audio_quality.estimated_mos
    
    def get_quality_summary(self) -> dict:
        """Get comprehensive quality summary."""
        mos = self.estimate_mos_score()
        
        return {
            "call_id": self.call_id,
            "duration_seconds": time.time() - self.start_time,
            "packet_loss_rate": round(self.packet_stats.loss_rate, 2),
            "packets_sent": self.packet_stats.sent,
            "packets_received": self.packet_stats.received,
            "packets_lost": self.packet_stats.lost,
            "jitter_ms": round(self.jitter_stats.current_jitter_ms, 2),
            "max_jitter_ms": round(self.jitter_stats.max_jitter_ms, 2),
            "buffer_underruns": self.buffer_stats.underruns,
            "buffer_overruns": self.buffer_stats.overruns,
            "buffer_health_score": round(self.buffer_stats.health_score, 1),
            "current_buffer_depth_ms": round(self.buffer_stats.current_depth_ms, 1),
            "estimated_mos": round(mos, 2),
            "avg_rtt_ms": round(sum(self.rtt_samples) / len(self.rtt_samples), 1) if self.rtt_samples else 0,
            "quality_rating": self._get_quality_rating(mos),
        }
    
    def _get_quality_rating(self, mos: float) -> str:
        """Convert MOS score to quality rating."""
        if mos >= 4.0:
            return "Excellent"
        elif mos >= 3.5:
            return "Good"
        elif mos >= 3.0:
            return "Fair"
        elif mos >= 2.0:
            return "Poor"
        else:
            return "Bad"
    
    async def start_monitoring(self):
        """Start periodic monitoring and logging."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"[{self.call_id}] Network monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring and log final summary."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Log final summary
        summary = self.get_quality_summary()
        logger.info(f"[{self.call_id}] Final network quality summary: {summary}")
    
    async def _monitor_loop(self):
        """Periodic monitoring loop."""
        try:
            while self._monitoring:
                await asyncio.sleep(self.config.packet_loss_check_interval_ms / 1000.0)
                
                if self.config.log_quality_metrics:
                    summary = self.get_quality_summary()
                    
                    # Log warnings if quality is degraded
                    if summary["packet_loss_rate"] > self.config.packet_loss_warning_threshold:
                        logger.warning(f"[{self.call_id}] Packet loss: {summary['packet_loss_rate']}%")
                    
                    if summary["jitter_ms"] > self.config.jitter_warning_threshold_ms:
                        logger.warning(f"[{self.call_id}] High jitter: {summary['jitter_ms']}ms")
                    
                    if summary["estimated_mos"] < 3.0:
                        logger.warning(f"[{self.call_id}] Poor audio quality: MOS={summary['estimated_mos']}")
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"[{self.call_id}] Monitor loop error: {e}")
