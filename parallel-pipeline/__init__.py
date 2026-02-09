"""
Parallel Order Accuracy Pipeline

A scalable multi-station order validation system with autoscaling.

Key Components:
- StationManager: Manages worker pool with autoscaling
- StationWorker: Per-station processing (GStreamer, OCR, YOLO, VLM)
- VLMScheduler: Batches VLM requests for GPU efficiency
- MetricsCollector: Tracks CPU/GPU/latency for scaling decisions
- ScalingPolicy: Implements scaling logic with hysteresis

Usage:
    from station_manager import StationManager
    from config import SystemConfig
    
    config = SystemConfig.from_yaml('config/system_config.yaml')
    manager = StationManager(config.to_dict())
    manager.start()
"""

__version__ = "1.0.0"
__author__ = "Order Accuracy Team"

from .station_manager import StationManager
from .station_worker import StationWorker, start_worker_process
from .vlm_scheduler import VLMScheduler
from .metrics_collector import MetricsCollector, MetricsStore
from .scaling_policy import ScalingPolicy, ScalingDecision, ScalingThresholds
from .shared_queue import QueueManager, VLMRequest, VLMResponse, SharedQueue, QueueBackend
from .config import SystemConfig, ScalingConfig, VLMConfig, StorageConfig

__all__ = [
    # Core components
    'StationManager',
    'StationWorker',
    'VLMScheduler',
    'MetricsCollector',
    'MetricsStore',
    
    # Scaling
    'ScalingPolicy',
    'ScalingDecision',
    'ScalingThresholds',
    
    # Queues
    'QueueManager',
    'VLMRequest',
    'VLMResponse',
    'SharedQueue',
    'QueueBackend',
    
    # Configuration
    'SystemConfig',
    'ScalingConfig',
    'VLMConfig',
    'StorageConfig',
    
    # Worker entry point
    'start_worker_process',
]
