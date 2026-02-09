# Parallel Order Accuracy Pipeline

Multi-station parallel processing architecture with autoscaling for order validation using camera streams.

## Overview

This refactored architecture transforms the sequential order validation pipeline into a parallel, scalable system that supports multiple camera stations with dynamic autoscaling based on resource utilization.

### Key Features

- **Multiprocessing Architecture**: Each station runs in separate process
- **Shared VLM Scheduler**: Batches inference requests from all stations
- **Autoscaling**: Dynamically adds/removes stations based on CPU, GPU, and latency
- **Graceful Shutdown**: Workers finish current orders before stopping
- **Benchmark Mode**: Measure maximum capacity and performance
- **Order Accuracy Preserved**: All business logic unchanged

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Station Manager                            │
│  - Manages worker pool                                          │
│  - Monitors metrics                                             │
│  - Applies scaling policy                                       │
└────────────────┬────────────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
┌───────▼───────┐  ┌──────▼───────┐  ...  ┌──────────────┐
│ Station       │  │ Station      │       │ Station      │
│ Worker 1      │  │ Worker 2     │       │ Worker N     │
│               │  │              │       │              │
│ - GStreamer   │  │ - GStreamer  │       │ - GStreamer  │
│ - OCR         │  │ - OCR        │       │ - OCR        │
│ - YOLO        │  │ - YOLO       │       │ - YOLO       │
└───────┬───────┘  └──────┬───────┘       └──────┬───────┘
        │                 │                       │
        └─────────────────┴───────────────────────┘
                          │
                ┌─────────▼──────────┐
                │  VLM Scheduler     │
                │  - Request batching │
                │  - Time windowing   │
                │  - Response routing │
                └─────────┬──────────┘
                          │
                ┌─────────▼──────────┐
                │   OVMS Service     │
                │   (VLM Model)      │
                └────────────────────┘
```

## Components

### 1. Station Worker (`station_worker.py`)

Each worker is a separate process handling one camera station:

- **Video Processing**: GStreamer pipeline for frame extraction
- **OCR**: Detects order IDs from frames
- **Frame Selection**: YOLO-based ranking
- **VLM Request**: Sends top frames to scheduler
- **Validation**: Matches VLM output with expected order

**Key Design**: Preserves all existing business logic from sequential implementation.

### 2. VLM Scheduler (`vlm_scheduler.py`)

Centralized inference request scheduler:

- **Request Collection**: Polls queue for requests from all workers
- **Time-Window Batching**: Accumulates requests for 50-100ms
- **Batch Processing**: Sends batched requests to OVMS
- **Response Routing**: Returns results to correct station worker

**Benefit**: Improves GPU utilization through OVMS continuous batching.

### 3. Station Manager (`station_manager.py`)

Orchestrates worker pool with autoscaling:

- **Worker Lifecycle**: Start/stop station processes
- **Metrics Monitoring**: CPU, GPU, latency tracking
- **Scaling Decisions**: Apply policy-based scaling
- **Graceful Shutdown**: Ensures orders complete before worker stops

### 4. Metrics Collector (`metrics_collector.py`)

Background thread collecting system metrics:

- **CPU**: Uses `psutil` for CPU utilization
- **GPU**: Pluggable interface (Intel GPU, placeholder)
- **Latency**: Rolling average per station
- **Queue Depths**: Monitor request backlog

### 5. Scaling Policy (`scaling_policy.py`)

Implements autoscaling logic:

**Scale UP** when ALL conditions met:
- GPU utilization < 85%
- CPU utilization < 80%
- Latency < 5 seconds

**Scale DOWN** when ANY condition met:
- GPU utilization > 95%
- CPU utilization > 90%
- Latency > 5 seconds

**Hysteresis**: 30-second window prevents rapid oscillations.

### 6. Shared Queue (`shared_queue.py`)

Inter-process communication:

- **VLM Request Queue**: Workers → Scheduler
- **Response Queues**: Scheduler → Workers (per-station)
- **Metrics Queue**: All → Metrics collector
- **Control Queue**: Manager → Workers (shutdown signals)

Supports both `multiprocessing.Queue` and Redis backends.

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
numpy
opencv-python
requests
psutil
pyyaml
redis  # Optional, for Redis backend
```

## Configuration

Configuration file: `config/system_config.yaml`

```yaml
# RTSP streams (one per potential station)
rtsp_urls:
  - rtsp://192.168.1.100:8554/station_1
  - rtsp://192.168.1.100:8554/station_2

# Model paths
yolo_model_path: ./models/yolo11n_openvino_model

# Data paths
inventory_path: ./config/inventory.json
orders_path: ./config/orders.json

# Queue backend
queue_backend: multiprocessing  # or "redis"
redis_host: localhost
redis_port: 6379

# VLM configuration
vlm:
  ovms_url: http://localhost:8000
  model_name: vlm
  batch_window_ms: 100
  max_batch_size: 16
  max_workers: 4

# Storage
storage:
  minio_endpoint: localhost:9000
  minio_bucket: orders

# Scaling policy
scaling:
  enabled: true
  scale_up_gpu_threshold: 85.0
  scale_up_cpu_threshold: 80.0
  scale_up_latency_threshold: 5.0
  scale_down_gpu_threshold: 95.0
  scale_down_cpu_threshold: 90.0
  scale_down_latency_threshold: 5.0
  hysteresis_window: 30.0
  min_stations: 1
  max_stations: 8
```

## Usage

### 1. Production Mode (with Autoscaling)

```python
from station_manager import StationManager
from vlm_scheduler import VLMScheduler
from config import SystemConfig

# Load configuration
config = SystemConfig.from_yaml('./config/system_config.yaml')

# Create station manager
manager = StationManager(
    config=config.to_dict(),
    initial_stations=1
)

# Create VLM scheduler
scheduler = VLMScheduler(
    queue_manager=manager.queue_manager,
    ovms_url=config.vlm.ovms_url,
    model_name=config.vlm.model_name
)

# Start scheduler
scheduler.start()

# Start manager (runs control loop)
manager.start()
```

### 2. Benchmark Mode

```python
from benchmark_runner import BenchmarkRunner, run_standard_benchmark
from config import SystemConfig

config = SystemConfig.from_yaml('./config/system_config.yaml')

# Run standard benchmark (1, 2, 4, 6, 8 stations)
results = run_standard_benchmark(
    config=config.to_dict(),
    output_dir='./benchmark_results'
)

# Or custom benchmark
runner = BenchmarkRunner(config.to_dict())
runner.add_scenario("custom", num_stations=4, duration_seconds=120)
results = runner.run_all()
runner.export_results()
```

### 3. Fixed Number of Stations (No Autoscaling)

```python
manager = StationManager(config, initial_stations=4)
manager.disable_autoscaling()
manager.start()
```

## Integration with Existing Code

### Step 1: Adapt Pipeline Runner

Modify `station_worker.py` to import your existing pipeline:

```python
# In _initialize_pipeline()
from application_service.pipeline_runner import PipelineRunner

self._pipeline_runner = PipelineRunner(
    rtsp_url=self.rtsp_url,
    frame_callback=self._handle_frame,
    minio_config=self.config['minio']
)
```

### Step 2: Integrate Frame Selector

```python
from frame_selector_service.frame_selector import FrameSelector

self._frame_selector = FrameSelector(
    yolo_model_path=self.config['yolo_model_path']
)

# In _select_best_frames()
return self._frame_selector.select_top_frames(frames, top_k=3)
```

### Step 3: Integrate Validation

```python
from application_service.validation_agent import ValidationAgent

self._validation_agent = ValidationAgent(
    inventory_path=self.config['inventory_path'],
    orders_path=self.config['orders_path']
)

# In _validate_order()
return self._validation_agent.validate(
    order_id=self._current_order_id,
    detected_items=detected_items
)
```

### Step 4: Update VLM Request Format

Modify `vlm_scheduler.py` `_send_to_ovms()` to match your OVMS API:

```python
def _send_to_ovms(self, request: VLMRequest) -> VLMResponse:
    # Adapt to your OVMS endpoint format
    payload = {
        # Your specific format here
    }
    
    response = requests.post(url, json=payload, timeout=30.0)
    
    # Parse your specific response format
    detected_items = self._parse_vlm_output(response.json())
    
    return VLMResponse(...)
```

## Benchmarking

Benchmark results are saved to:
- `benchmark_results/benchmark_results_TIMESTAMP.json` - Full data
- `benchmark_results/benchmark_results_TIMESTAMP.csv` - Summary table

### Metrics Collected

- **Throughput**: Orders per second
- **Latency**: Average, P95, P99
- **Resource Usage**: CPU/GPU avg and max
- **Success Rate**: Percentage of successful validations

### Example Results

```
Scenario: scale_4_stations
Stations: 4
Duration: 120.0s

Throughput:
  Total Orders: 480
  Success Rate: 98.5%
  Orders/sec: 4.0

Latency:
  Average: 4.2s
  P95: 4.8s
  P99: 5.1s

Resource Usage:
  CPU: avg=72.3% max=85.1%
  GPU: avg=88.5% max=94.2%
```

## Testing

### Unit Tests

```python
# Test scaling policy
from scaling_policy import ScalingPolicy, ScalingDecision

policy = ScalingPolicy()
decision = policy.evaluate(
    current_stations=2,
    cpu_utilization=60.0,
    gpu_utilization=70.0,
    avg_latency=3.5
)

assert decision.action == ScalingDecision.SCALE_UP
```

### Simulation Mode

Replace real RTSP streams with prerecorded videos:

```python
config.rtsp_urls = [
    'file:///path/to/test_video_1.mp4',
    'file:///path/to/test_video_2.mp4'
]
```

## Performance Tuning

### VLM Batching

- **batch_window_ms**: Lower = faster individual requests, higher = better batching
  - Recommended: 50-100ms
- **max_batch_size**: Match OVMS max_num_seqs
  - Recommended: 16

### Scaling Policy

- **Conservative** (stability): Lower GPU threshold (70%), higher hysteresis (60s)
- **Aggressive** (throughput): Higher GPU threshold (90%), lower hysteresis (20s)

### Queue Backend

- **multiprocessing.Queue**: Fastest, single-node only
- **Redis**: Slightly slower, supports distributed deployment

## Monitoring

### Real-time Status

```python
status = manager.get_status()
print(f"Active stations: {status['active_stations']}")
print(f"CPU: {status['metrics']['cpu_avg']:.1f}%")
print(f"GPU: {status['metrics']['gpu_avg']:.1f}%")
print(f"Latency: {status['metrics']['latency_avg']:.2f}s")
```

### Scaling History

```python
history = manager.get_scaling_history()
for event in history:
    print(f"{event['action']}: {event['from']} → {event['to']} stations")
    print(f"  Reason: {event['reason']}")
```

## Troubleshooting

### High Latency

- Check GPU utilization - if low, increase batch window
- Check queue depths - if high, scale up workers
- Check OVMS response time

### Low GPU Utilization

- Increase batch window (100ms → 200ms)
- Increase max batch size
- Add more stations (if under max)

### Workers Not Scaling

- Check hysteresis window - may be preventing scaling
- Verify metrics are being collected
- Check scaling policy thresholds

## Safety Guarantees

### Order Accuracy

✅ **Business logic unchanged** - All validation logic preserved from original implementation

✅ **Deterministic processing** - Each worker processes orders independently

✅ **Request-response pairing** - VLM responses matched to correct requests via request_id

✅ **Graceful shutdown** - Workers finish current order before stopping

### Resource Protection

✅ **Scale down on overload** - Prevents CPU/GPU saturation

✅ **Hysteresis** - Avoids rapid scaling oscillations

✅ **Max stations limit** - Prevents resource exhaustion

## Future Enhancements

1. **Distributed Deployment**: Use Redis queue backend + Kubernetes
2. **Advanced Batching**: Implement true multi-request batching for OVMS
3. **Dynamic RTSP**: Hot-add/remove camera streams
4. **ML-based Scaling**: Predictive autoscaling using historical patterns
5. **A/B Testing**: Compare sequential vs parallel accuracy

## License

[Your License]

## Support

For issues or questions, contact [your contact info]
