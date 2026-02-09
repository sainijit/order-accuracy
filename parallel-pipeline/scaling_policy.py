"""
Autoscaling Policy Engine

Implements scaling decisions based on resource utilization and latency metrics.
Includes hysteresis to prevent rapid scaling oscillations.
"""

import logging
from typing import Tuple, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class ScalingThresholds:
    """Configuration for scaling thresholds"""
    
    # Scale UP thresholds (all conditions must be met)
    scale_up_gpu_threshold: float = 85.0  # GPU < 85%
    scale_up_cpu_threshold: float = 80.0  # CPU < 80%
    scale_up_latency_threshold: float = 5.0  # Latency < 5s
    
    # Scale DOWN thresholds (any condition triggers)
    scale_down_gpu_threshold: float = 95.0  # GPU > 95%
    scale_down_cpu_threshold: float = 90.0  # CPU > 90%
    scale_down_latency_threshold: float = 5.0  # Latency > 5s
    
    # Hysteresis settings
    hysteresis_window: float = 30.0  # Seconds between scaling actions
    min_stations: int = 1
    max_stations: int = 8
    
    # Conservative scaling
    scale_up_increment: int = 1  # Add 1 station at a time
    scale_down_increment: int = 1  # Remove 1 station at a time


class ScalingDecision:
    """Represents a scaling decision"""
    
    NONE = "none"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    
    def __init__(
        self,
        action: str,
        target_stations: int,
        reason: str,
        metrics: dict
    ):
        self.action = action
        self.target_stations = target_stations
        self.reason = reason
        self.metrics = metrics
        self.timestamp = time.time()
    
    def __repr__(self):
        return (
            f"ScalingDecision(action={self.action}, "
            f"target={self.target_stations}, reason={self.reason})"
        )


class ScalingPolicy:
    """
    Implements autoscaling logic based on resource utilization.
    
    Scale UP Policy:
        Conditions (ALL must be true):
        - GPU utilization < 85%
        - CPU utilization < 80%
        - Average latency < 5 seconds
        - No scaling action in last 30 seconds (hysteresis)
        
        Action: Add 1 station
    
    Scale DOWN Policy:
        Conditions (ANY must be true):
        - GPU utilization > 95%
        - CPU utilization > 90%
        - Average latency > 5 seconds
        - No scaling action in last 30 seconds (hysteresis)
        
        Action: Remove 1 station
    
    Hysteresis prevents rapid scaling oscillations.
    """
    
    def __init__(self, thresholds: Optional[ScalingThresholds] = None):
        """
        Args:
            thresholds: Scaling threshold configuration
        """
        self.thresholds = thresholds or ScalingThresholds()
        self._last_scaling_time = 0.0
        self._scaling_history = []
        
        logger.info(
            f"ScalingPolicy initialized: "
            f"UP[GPU<{self.thresholds.scale_up_gpu_threshold}% "
            f"CPU<{self.thresholds.scale_up_cpu_threshold}% "
            f"LAT<{self.thresholds.scale_up_latency_threshold}s] "
            f"DOWN[GPU>{self.thresholds.scale_down_gpu_threshold}% "
            f"CPU>{self.thresholds.scale_down_cpu_threshold}% "
            f"LAT>{self.thresholds.scale_down_latency_threshold}s]"
        )
    
    def evaluate(
        self,
        current_stations: int,
        cpu_utilization: float,
        gpu_utilization: float,
        avg_latency: float
    ) -> ScalingDecision:
        """
        Evaluate scaling decision based on current metrics.
        
        Args:
            current_stations: Number of currently active stations
            cpu_utilization: Average CPU utilization (0-100)
            gpu_utilization: Average GPU utilization (0-100)
            avg_latency: Average order latency in seconds
        
        Returns:
            ScalingDecision with action and reasoning
        """
        metrics = {
            'cpu': cpu_utilization,
            'gpu': gpu_utilization,
            'latency': avg_latency,
            'stations': current_stations
        }
        
        # Check hysteresis window
        time_since_last_scaling = time.time() - self._last_scaling_time
        if time_since_last_scaling < self.thresholds.hysteresis_window:
            return ScalingDecision(
                action=ScalingDecision.NONE,
                target_stations=current_stations,
                reason=f"Hysteresis: {time_since_last_scaling:.1f}s < {self.thresholds.hysteresis_window}s",
                metrics=metrics
            )
        
        # Evaluate SCALE DOWN conditions (priority: prevent overload)
        scale_down_conditions = []
        
        if gpu_utilization > self.thresholds.scale_down_gpu_threshold:
            scale_down_conditions.append(
                f"GPU overload ({gpu_utilization:.1f}% > {self.thresholds.scale_down_gpu_threshold}%)"
            )
        
        if cpu_utilization > self.thresholds.scale_down_cpu_threshold:
            scale_down_conditions.append(
                f"CPU overload ({cpu_utilization:.1f}% > {self.thresholds.scale_down_cpu_threshold}%)"
            )
        
        if avg_latency > self.thresholds.scale_down_latency_threshold:
            scale_down_conditions.append(
                f"Latency violation ({avg_latency:.2f}s > {self.thresholds.scale_down_latency_threshold}s)"
            )
        
        # If any scale down condition triggered
        if scale_down_conditions and current_stations > self.thresholds.min_stations:
            target = max(
                current_stations - self.thresholds.scale_down_increment,
                self.thresholds.min_stations
            )
            
            reason = "Scale DOWN: " + "; ".join(scale_down_conditions)
            
            self._last_scaling_time = time.time()
            self._scaling_history.append(('down', time.time(), metrics))
            
            logger.info(
                f"SCALING DECISION: DOWN from {current_stations} to {target} stations. "
                f"Reason: {reason}"
            )
            
            return ScalingDecision(
                action=ScalingDecision.SCALE_DOWN,
                target_stations=target,
                reason=reason,
                metrics=metrics
            )
        
        # Evaluate SCALE UP conditions (all must be true)
        can_scale_up = (
            current_stations < self.thresholds.max_stations and
            gpu_utilization < self.thresholds.scale_up_gpu_threshold and
            cpu_utilization < self.thresholds.scale_up_cpu_threshold and
            avg_latency < self.thresholds.scale_up_latency_threshold
        )
        
        if can_scale_up:
            target = min(
                current_stations + self.thresholds.scale_up_increment,
                self.thresholds.max_stations
            )
            
            reason = (
                f"Scale UP: GPU={gpu_utilization:.1f}% < {self.thresholds.scale_up_gpu_threshold}%, "
                f"CPU={cpu_utilization:.1f}% < {self.thresholds.scale_up_cpu_threshold}%, "
                f"Latency={avg_latency:.2f}s < {self.thresholds.scale_up_latency_threshold}s"
            )
            
            self._last_scaling_time = time.time()
            self._scaling_history.append(('up', time.time(), metrics))
            
            logger.info(
                f"SCALING DECISION: UP from {current_stations} to {target} stations. "
                f"Reason: {reason}"
            )
            
            return ScalingDecision(
                action=ScalingDecision.SCALE_UP,
                target_stations=target,
                reason=reason,
                metrics=metrics
            )
        
        # No scaling action needed
        return ScalingDecision(
            action=ScalingDecision.NONE,
            target_stations=current_stations,
            reason=(
                f"Stable: GPU={gpu_utilization:.1f}%, "
                f"CPU={cpu_utilization:.1f}%, "
                f"Latency={avg_latency:.2f}s"
            ),
            metrics=metrics
        )
    
    def get_scaling_history(self) -> list:
        """Get history of scaling actions"""
        return self._scaling_history.copy()
    
    def reset_hysteresis(self):
        """Reset hysteresis timer (useful for testing)"""
        self._last_scaling_time = 0.0
        logger.debug("Hysteresis timer reset")


class ConservativeScalingPolicy(ScalingPolicy):
    """
    More conservative scaling policy for production.
    
    - Higher thresholds for scale up
    - More aggressive scale down
    - Longer hysteresis window
    """
    
    def __init__(self):
        thresholds = ScalingThresholds(
            scale_up_gpu_threshold=70.0,  # More headroom before scaling up
            scale_up_cpu_threshold=70.0,
            scale_up_latency_threshold=4.0,  # Lower latency threshold
            
            scale_down_gpu_threshold=95.0,
            scale_down_cpu_threshold=90.0,
            scale_down_latency_threshold=5.5,  # Slightly higher tolerance
            
            hysteresis_window=60.0,  # Longer wait between changes
            min_stations=1,
            max_stations=6  # Conservative max
        )
        
        super().__init__(thresholds)
        logger.info("ConservativeScalingPolicy initialized")


class AggressiveScalingPolicy(ScalingPolicy):
    """
    Aggressive scaling policy for maximum throughput.
    
    - Lower thresholds for scale up
    - Tolerates higher resource usage
    - Shorter hysteresis window
    """
    
    def __init__(self):
        thresholds = ScalingThresholds(
            scale_up_gpu_threshold=90.0,  # Push GPU harder
            scale_up_cpu_threshold=85.0,
            scale_up_latency_threshold=5.0,
            
            scale_down_gpu_threshold=98.0,  # Only scale down at extreme load
            scale_down_cpu_threshold=95.0,
            scale_down_latency_threshold=6.0,
            
            hysteresis_window=20.0,  # Faster reactions
            min_stations=2,  # Start with more capacity
            max_stations=8
        )
        
        super().__init__(thresholds)
        logger.info("AggressiveScalingPolicy initialized")
