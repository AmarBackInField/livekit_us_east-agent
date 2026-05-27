"""
Redis Manager for Distributed State Management

Handles call state caching, pub/sub coordination, and session recovery
for production telephony deployments with 100-500 concurrent calls.
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from config import RedisConfig

logger = logging.getLogger("redis-manager")


class RedisManager:
    """
    Manages Redis connections and operations for distributed agent state.
    
    Features:
    - Connection pooling for efficiency
    - Call state caching with TTL
    - Pub/sub for event coordination
    - Session recovery support
    - Rate limiting and circuit breakers
    """
    
    def __init__(self, config: Optional[RedisConfig] = None):
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - install with: pip install redis")
            self._enabled = False
            return
        
        self.config = config or RedisConfig()
        self._enabled = True
        self._pool: Optional[redis.ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._subscribers: Dict[str, list] = {}
        
        logger.info("Redis manager initialized")
    
    async def connect(self):
        """Initialize Redis connection pool."""
        if not self._enabled:
            return
        
        try:
            self._pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=True,
            )
            
            self._client = redis.Redis(connection_pool=self._pool)
            
            # Test connection
            await self._client.ping()
            logger.info(f"Redis connected: {self.config.host}:{self.config.port}")
            
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self._enabled = False
    
    async def disconnect(self):
        """Close Redis connections."""
        if not self._enabled or not self._client:
            return
        
        try:
            if self._pubsub:
                await self._pubsub.close()
            await self._client.close()
            if self._pool:
                await self._pool.disconnect()
            logger.info("Redis disconnected")
        except Exception as e:
            logger.error(f"Redis disconnect error: {e}")
    
    # =========================================================================
    # Call State Management
    # =========================================================================
    
    async def save_call_state(self, call_id: str, state: Dict[str, Any]) -> bool:
        """
        Save call state to Redis with TTL.
        
        Args:
            call_id: Unique call identifier
            state: Call state dictionary
        
        Returns:
            True if saved successfully
        """
        if not self._enabled or not self._client:
            return False
        
        try:
            key = f"call:{call_id}"
            value = json.dumps(state, default=str)
            await self._client.setex(key, self.config.call_state_ttl, value)
            logger.debug(f"Saved call state: {call_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save call state {call_id}: {e}")
            return False
    
    async def get_call_state(self, call_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve call state from Redis.
        
        Args:
            call_id: Unique call identifier
        
        Returns:
            Call state dictionary or None if not found
        """
        if not self._enabled or not self._client:
            return None
        
        try:
            key = f"call:{call_id}"
            value = await self._client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Failed to get call state {call_id}: {e}")
            return None
    
    async def delete_call_state(self, call_id: str) -> bool:
        """Delete call state from Redis."""
        if not self._enabled or not self._client:
            return False
        
        try:
            key = f"call:{call_id}"
            await self._client.delete(key)
            logger.debug(f"Deleted call state: {call_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete call state {call_id}: {e}")
            return False
    
    async def update_call_metrics(self, call_id: str, metrics: Dict[str, Any]) -> bool:
        """Update specific metrics for a call."""
        if not self._enabled or not self._client:
            return False
        
        try:
            key = f"metrics:{call_id}"
            value = json.dumps(metrics, default=str)
            await self._client.setex(key, self.config.metrics_ttl, value)
            return True
        except Exception as e:
            logger.error(f"Failed to update metrics {call_id}: {e}")
            return False
    
    # =========================================================================
    # Session Caching
    # =========================================================================
    
    async def cache_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Cache session data with shorter TTL."""
        if not self._enabled or not self._client:
            return False
        
        try:
            key = f"session:{session_id}"
            value = json.dumps(data, default=str)
            await self._client.setex(key, self.config.session_cache_ttl, value)
            return True
        except Exception as e:
            logger.error(f"Failed to cache session {session_id}: {e}")
            return False
    
    async def get_cached_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached session data."""
        if not self._enabled or not self._client:
            return None
        
        try:
            key = f"session:{session_id}"
            value = await self._client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    # =========================================================================
    # Pub/Sub for Event Coordination
    # =========================================================================
    
    async def publish_event(self, event_type: str, data: Dict[str, Any]) -> bool:
        """
        Publish an event to Redis pub/sub.
        
        Args:
            event_type: Type of event (e.g., 'call_started', 'call_ended')
            data: Event data
        
        Returns:
            True if published successfully
        """
        if not self._enabled or not self._client:
            return False
        
        try:
            message = json.dumps({
                "type": event_type,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, default=str)
            
            await self._client.publish(self.config.events_channel, message)
            logger.debug(f"Published event: {event_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish event {event_type}: {e}")
            return False
    
    async def publish_metric(self, metric_name: str, value: Any, tags: Optional[Dict] = None) -> bool:
        """Publish a metric to the metrics channel."""
        if not self._enabled or not self._client:
            return False
        
        try:
            message = json.dumps({
                "metric": metric_name,
                "value": value,
                "tags": tags or {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, default=str)
            
            await self._client.publish(self.config.metrics_channel, message)
            return True
        except Exception as e:
            logger.error(f"Failed to publish metric {metric_name}: {e}")
            return False
    
    async def publish_alert(self, alert_type: str, message: str, severity: str = "warning") -> bool:
        """Publish an alert to the alerts channel."""
        if not self._enabled or not self._client:
            return False
        
        try:
            alert_data = json.dumps({
                "type": alert_type,
                "message": message,
                "severity": severity,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            
            await self._client.publish(self.config.alerts_channel, alert_data)
            logger.info(f"Alert published: {alert_type} - {message}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish alert: {e}")
            return False
    
    # =========================================================================
    # Rate Limiting
    # =========================================================================
    
    async def check_rate_limit(self, key: str, limit: int, window_seconds: int) -> bool:
        """
        Check if rate limit is exceeded using sliding window.
        
        Args:
            key: Rate limit key (e.g., 'calls:agent_id')
            limit: Maximum number of requests
            window_seconds: Time window in seconds
        
        Returns:
            True if within limit, False if exceeded
        """
        if not self._enabled or not self._client:
            return True  # Allow if Redis unavailable
        
        try:
            now = datetime.now(timezone.utc).timestamp()
            window_start = now - window_seconds
            
            # Use sorted set for sliding window
            rate_key = f"rate:{key}"
            
            # Remove old entries
            await self._client.zremrangebyscore(rate_key, 0, window_start)
            
            # Count current entries
            count = await self._client.zcard(rate_key)
            
            if count >= limit:
                return False
            
            # Add current request
            await self._client.zadd(rate_key, {str(now): now})
            await self._client.expire(rate_key, window_seconds)
            
            return True
        except Exception as e:
            logger.error(f"Rate limit check failed for {key}: {e}")
            return True  # Allow on error
    
    # =========================================================================
    # Circuit Breaker
    # =========================================================================
    
    async def record_failure(self, service: str) -> int:
        """Record a service failure and return failure count."""
        if not self._enabled or not self._client:
            return 0
        
        try:
            key = f"circuit:{service}:failures"
            count = await self._client.incr(key)
            await self._client.expire(key, 60)  # Reset after 1 minute
            return count
        except Exception as e:
            logger.error(f"Failed to record failure for {service}: {e}")
            return 0
    
    async def record_success(self, service: str):
        """Record a service success and reset failure count."""
        if not self._enabled or not self._client:
            return
        
        try:
            key = f"circuit:{service}:failures"
            await self._client.delete(key)
        except Exception as e:
            logger.error(f"Failed to record success for {service}: {e}")
    
    async def is_circuit_open(self, service: str, threshold: int = 5) -> bool:
        """Check if circuit breaker is open (too many failures)."""
        if not self._enabled or not self._client:
            return False
        
        try:
            key = f"circuit:{service}:failures"
            count = await self._client.get(key)
            return int(count or 0) >= threshold
        except Exception as e:
            logger.error(f"Failed to check circuit for {service}: {e}")
            return False
    
    # =========================================================================
    # Health Check
    # =========================================================================
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Redis health and return status."""
        if not self._enabled:
            return {"status": "disabled", "available": False}
        
        try:
            if not self._client:
                return {"status": "not_connected", "available": False}
            
            # Ping test
            start = asyncio.get_event_loop().time()
            await self._client.ping()
            latency_ms = (asyncio.get_event_loop().time() - start) * 1000
            
            # Get info
            info = await self._client.info()
            
            return {
                "status": "healthy",
                "available": True,
                "latency_ms": round(latency_ms, 2),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "available": False, "error": str(e)}


# Global Redis manager instance
_redis_manager: Optional[RedisManager] = None


async def get_redis_manager() -> RedisManager:
    """Get or create global Redis manager instance."""
    global _redis_manager
    if _redis_manager is None:
        _redis_manager = RedisManager()
        await _redis_manager.connect()
    return _redis_manager
