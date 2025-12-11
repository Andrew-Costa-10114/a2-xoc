"""Performance monitoring and metrics collection for the miner API.

This module provides comprehensive performance tracking including:
- Component execution times
- Token usage statistics
- Error rates and success/failure tracking
- Request throughput and latency
- Performance percentiles
"""

import time
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance metric entry."""
    component: str
    execution_time_ms: float
    tokens_used: int = 0
    success: bool = True
    error_type: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    finish_reason: Optional[str] = None
    is_evaluation: bool = False  # Track if this is an evaluation request


@dataclass
class ComponentStats:
    """Aggregated statistics for a component."""
    component: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_execution_time_ms: float = 0.0
    min_execution_time_ms: float = float('inf')
    max_execution_time_ms: float = 0.0
    recent_execution_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    @property
    def avg_execution_time_ms(self) -> float:
        """Calculate average execution time."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_execution_time_ms / self.successful_requests
    
    @property
    def avg_tokens_per_request(self) -> float:
        """Calculate average tokens per request."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_tokens / self.successful_requests
    
    @property
    def p50_execution_time_ms(self) -> float:
        """Calculate 50th percentile (median) execution time."""
        if not self.recent_execution_times:
            return 0.0
        sorted_times = sorted(self.recent_execution_times)
        idx = len(sorted_times) // 2
        return sorted_times[idx]
    
    @property
    def p95_execution_time_ms(self) -> float:
        """Calculate 95th percentile execution time."""
        if not self.recent_execution_times:
            return 0.0
        sorted_times = sorted(self.recent_execution_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[idx] if idx < len(sorted_times) else sorted_times[-1]
    
    @property
    def p99_execution_time_ms(self) -> float:
        """Calculate 99th percentile execution time."""
        if not self.recent_execution_times:
            return 0.0
        sorted_times = sorted(self.recent_execution_times)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[idx] if idx < len(sorted_times) else sorted_times[-1]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for JSON serialization."""
        return {
            "component": self.component,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(self.success_rate, 2),
            "total_tokens": self.total_tokens,
            "avg_tokens_per_request": round(self.avg_tokens_per_request, 2),
            "total_execution_time_ms": round(self.total_execution_time_ms, 2),
            "avg_execution_time_ms": round(self.avg_execution_time_ms, 2),
            "min_execution_time_ms": round(self.min_execution_time_ms, 2) if self.min_execution_time_ms != float('inf') else 0.0,
            "max_execution_time_ms": round(self.max_execution_time_ms, 2),
            "p50_execution_time_ms": round(self.p50_execution_time_ms, 2),
            "p95_execution_time_ms": round(self.p95_execution_time_ms, 2),
            "p99_execution_time_ms": round(self.p99_execution_time_ms, 2),
        }


class PerformanceMonitor:
    """Thread-safe performance monitoring system."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            max_history: Maximum number of metrics to keep in history
        """
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.component_stats: Dict[str, ComponentStats] = defaultdict(lambda: ComponentStats(component=""))
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.lock = Lock()
        self.start_time = datetime.utcnow()
        self.total_requests = 0
        self.total_successful_requests = 0
        self.total_failed_requests = 0
        self.total_tokens = 0
        # Evaluation-round response time tracking (for minimum response time enforcement)
        self._evaluation_round_times: deque = deque(maxlen=100)  # Last 100 evaluation response times
    
    @property
    def evaluation_round_times(self) -> deque:
        """Get evaluation round times deque (thread-safe access)."""
        with self.lock:
            return self._evaluation_round_times
    
    def record_metric(self, metric: PerformanceMetric):
        """
        Record a performance metric.
        
        Args:
            metric: Performance metric to record
        """
        with self.lock:
            # Update component stats
            stats = self.component_stats[metric.component]
            stats.component = metric.component
            stats.total_requests += 1
            self.total_requests += 1
            
            if metric.success:
                stats.successful_requests += 1
                self.total_successful_requests += 1
                stats.total_execution_time_ms += metric.execution_time_ms
                stats.min_execution_time_ms = min(stats.min_execution_time_ms, metric.execution_time_ms)
                stats.max_execution_time_ms = max(stats.max_execution_time_ms, metric.execution_time_ms)
                stats.recent_execution_times.append(metric.execution_time_ms)
                stats.total_tokens += metric.tokens_used
                self.total_tokens += metric.tokens_used
            else:
                stats.failed_requests += 1
                self.total_failed_requests += 1
                if metric.error_type:
                    self.error_counts[metric.error_type] += 1
            
            # Track evaluation-round response times separately
            if metric.is_evaluation and metric.success:
                self._evaluation_round_times.append(metric.execution_time_ms)
            
            # Add to history
            self.metrics_history.append(metric)
    
    def get_component_stats(self, component: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for a component or all components.
        
        Args:
            component: Component name (None for all components)
            
        Returns:
            Dictionary with component statistics
        """
        with self.lock:
            if component:
                stats = self.component_stats.get(component)
                return stats.to_dict() if stats else {}
            else:
                return {
                    comp: stats.to_dict()
                    for comp, stats in self.component_stats.items()
                }
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """
        Get overall performance statistics.
        
        Returns:
            Dictionary with overall statistics
        """
        with self.lock:
            uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
            uptime_minutes = uptime_seconds / 60.0
            
            return {
                "uptime_seconds": round(uptime_seconds, 2),
                "uptime_minutes": round(uptime_minutes, 2),
                "start_time": self.start_time.isoformat(),
                "total_requests": self.total_requests,
                "total_successful_requests": self.total_successful_requests,
                "total_failed_requests": self.total_failed_requests,
                "overall_success_rate": round((self.total_successful_requests / self.total_requests * 100.0) if self.total_requests > 0 else 100.0, 2),
                "total_tokens": self.total_tokens,
                "avg_tokens_per_request": round(self.total_tokens / self.total_requests if self.total_requests > 0 else 0, 2),
                "requests_per_minute": round(self.total_requests / uptime_minutes if uptime_minutes > 0 else 0, 2),
                "successful_requests_per_minute": round(self.total_successful_requests / uptime_minutes if uptime_minutes > 0 else 0, 2),
                "error_counts": dict(self.error_counts),
                "components": list(self.component_stats.keys()),
            }
    
    def get_recent_metrics(self, limit: int = 100, component: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent metrics.
        
        Args:
            limit: Maximum number of metrics to return
            component: Filter by component name (None for all)
            
        Returns:
            List of recent metrics as dictionaries
        """
        with self.lock:
            metrics = list(self.metrics_history)
            if component:
                metrics = [m for m in metrics if m.component == component]
            metrics = metrics[-limit:]
            
            return [
                {
                    "component": m.component,
                    "execution_time_ms": round(m.execution_time_ms, 2),
                    "tokens_used": m.tokens_used,
                    "success": m.success,
                    "error_type": m.error_type,
                    "timestamp": m.timestamp.isoformat(),
                    "finish_reason": m.finish_reason,
                }
                for m in metrics
            ]
    
    def get_evaluation_round_avg_time_ms(self) -> float:
        """
        Get average evaluation-round response time.
        
        Returns:
            Average response time in milliseconds for evaluation rounds.
            Returns 0.0 if no evaluation data available.
        """
        with self.lock:
            if not self._evaluation_round_times:
                return 0.0
            return sum(self._evaluation_round_times) / len(self._evaluation_round_times)
    
    def get_minimum_valid_response_time_ms(self) -> float:
        """
        Calculate minimum valid response time (1/3 of evaluation-round average).
        
        This is used to enforce minimum response time to prevent gaming.
        
        Returns:
            Minimum valid response time in milliseconds (1/3 of evaluation-round average).
            Returns 0.0 if no evaluation data available.
        """
        avg_time = self.get_evaluation_round_avg_time_ms()
        if avg_time == 0.0:
            return 0.0
        return avg_time / 3.0
    
    def reset_stats(self):
        """Reset all statistics (useful for testing)."""
        with self.lock:
            self.metrics_history.clear()
            self.component_stats.clear()
            self.error_counts.clear()
            self._evaluation_round_times.clear()
            self.start_time = datetime.utcnow()
            self.total_requests = 0
            self.total_successful_requests = 0
            self.total_failed_requests = 0
            self.total_tokens = 0


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


class PerformanceTracker:
    """Context manager for tracking component performance."""
    
    def __init__(self, component: str, is_evaluation: bool = False):
        """
        Initialize performance tracker.
        
        Args:
            component: Component name
            is_evaluation: Whether this is an evaluation request (for response time enforcement)
        """
        self.component = component
        self.start_time: Optional[float] = None
        self.tokens_used: int = 0
        self.success: bool = True
        self.error_type: Optional[str] = None
        self.finish_reason: Optional[str] = None
        self.is_evaluation = is_evaluation
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Record metric on exit."""
        if self.start_time is None:
            return
        
        execution_time_ms = (time.perf_counter() - self.start_time) * 1000.0
        
        if exc_type is not None:
            self.success = False
            self.error_type = f"{exc_type.__name__}: {str(exc_val)}"
        
        metric = PerformanceMetric(
            component=self.component,
            execution_time_ms=execution_time_ms,
            tokens_used=self.tokens_used,
            success=self.success,
            error_type=self.error_type,
            finish_reason=self.finish_reason,
            is_evaluation=self.is_evaluation,
        )
        
        performance_monitor.record_metric(metric)
        
        # Log performance
        if self.success:
            logger.info(
                f"[{self.component}] Execution: {execution_time_ms:.2f}ms, "
                f"Tokens: {self.tokens_used}, Finish: {self.finish_reason or 'N/A'}"
            )
        else:
            logger.warning(
                f"[{self.component}] Failed: {execution_time_ms:.2f}ms, "
                f"Error: {self.error_type}"
            )
        
        return False  # Don't suppress exceptions
    
    def set_tokens(self, tokens: int):
        """Set token usage."""
        self.tokens_used = tokens
    
    def set_finish_reason(self, reason: str):
        """Set finish reason."""
        self.finish_reason = reason


def track_performance(component: str):
    """
    Decorator to track performance of a function.
    
    Args:
        component: Component name for tracking
        
    Usage:
        @track_performance("complete")
        async def component_complete(...):
            ...
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                with PerformanceTracker(component) as tracker:
                    try:
                        result = await func(*args, **kwargs)
                        # Try to extract token usage from result if available
                        if hasattr(result, 'output') and hasattr(result.output, 'immediate_response'):
                            # Token usage will be set manually in components
                            pass
                        return result
                    except Exception as e:
                        tracker.success = False
                        tracker.error_type = f"{type(e).__name__}: {str(e)}"
                        raise
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                with PerformanceTracker(component) as tracker:
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception as e:
                        tracker.success = False
                        tracker.error_type = f"{type(e).__name__}: {str(e)}"
                        raise
            return sync_wrapper
    return decorator

