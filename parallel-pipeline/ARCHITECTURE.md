# Parallel Order Accuracy Pipeline - Architecture

## Overview

This is a parallel processing system for real-time order accuracy validation across multiple camera stations. The system is designed to handle 6-8 concurrent camera streams with GPU-accelerated VLM inference and automatic scaling.

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Station 1      │────▶│                 │────▶│                 │
│  Worker         │     │                 │     │                 │
└─────────────────┘     │                 │     │   VLM           │
                        │  VLM Scheduler  │────▶│   Service       │
┌─────────────────┐     │  (Batching)     │     │   (OVMS)        │
│  Station 2      │────▶│                 │     │                 │
│  Worker         │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                        │
        │                       │                        │
┌───────▼───────────────────────▼────────────────────────▼─────────┐
│                    Shared Infrastructure                          │
│  • Multiprocessing Queues (Request/Response)                      │
│  • Metrics Store (CPU, GPU, Latency)                              │
│  • MinIO Object Storage (Frame Storage)                           │
│  • Semantic Comparison Service (AI Matching)                      │
└───────────────────────────────────────────────────────────────────┘
        │                       │                        │
        ▼                       ▼                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Station        │     │  Scaling        │     │  Metrics        │
│  Manager        │────▶│  Policy         │◀────│  Collector      │
│  (Autoscaling)  │     │  (Conservative) │     │  (Real-time)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Core Components

### 1. Station Worker (`station_worker.py`)
**Purpose**: Handles complete order processing for one camera station

**Responsibilities**:
- Video ingestion via GStreamer pipeline
- Frame extraction and OCR for order ID detection
- Frame upload to MinIO storage
- Frame selection using YOLO (top-3 scoring)
- VLM inference request submission
- Order validation with semantic matching
- Metrics reporting

**Process Isolation**: Each worker runs in separate process for fault isolation

**Key Methods**:
- `_run_gstreamer_pipeline()`: Start video processing
- `_poll_frames_from_storage()`: Load frames from MinIO
- `_select_best_frames()`: YOLO-based frame ranking
- `_request_vlm_inference()`: Submit to VLM scheduler
- `_validate_order()`: Compare detected vs expected items

---

### 2. VLM Scheduler (`vlm_scheduler.py`)
**Purpose**: Batch VLM inference requests for optimal GPU utilization

**Batching Strategy**:
- Time window: 100ms (configurable)
- Max batch size: 16 requests
- Fair scheduling across stations

**Responsibilities**:
- Collect requests from all station workers
- Create time-windowed batches
- Send batches to OVMS VLM service
- Route responses back to station workers
- Error handling and retry logic

**Threading Model**:
- 1 collector thread: Polls request queue
- 1 batcher thread: Creates time-windowed batches
- 4 worker threads: Process batches in parallel

**Key Optimizations**:
- Small time windows prevent excessive latency
- OVMS handles internal GPU batching
- Fair station scheduling prevents starvation

---

### 3. Station Manager (`station_manager.py`)
**Purpose**: Manage worker pool with dynamic autoscaling

**Autoscaling Logic**:
- Scale up when: GPU > 85%, CPU > 80%, or latency > 5s
- Scale down when: GPU < 40%, CPU < 30%, and latency < 2s
- Hysteresis window: 30s (prevent oscillation)
- Range: 1-8 stations

**Responsibilities**:
- Start/stop station worker processes
- Monitor system metrics
- Apply scaling policy decisions
- Graceful worker lifecycle management
- Queue depth monitoring

**Control Loop**:
- Interval: 5 seconds
- Checks: CPU, GPU, latency metrics
- Actions: Scale up, scale down, or hold steady

---

### 4. Shared Queue Manager (`shared_queue.py`)
**Purpose**: IPC between processes using multiprocessing queues

**Queue Types**:
- `vlm_request_queue`: Station workers → VLM scheduler
- `vlm_response_queue_{station_id}`: VLM scheduler → Station worker
- `control_queue`: Station manager → Workers (shutdown signals)

**Data Structures**:
- `VLMRequest`: Station ID, order ID, frames, timestamp
- `VLMResponse`: Request ID, detected items, inference time, success flag

**Backend Support**:
- Multiprocessing (default): Uses Python's multiprocessing.Queue
- Redis (optional): For distributed deployment

---

### 5. Metrics Collector (`metrics_collector.py`)
**Purpose**: Track system performance metrics in real-time

**Metrics Tracked**:
- CPU utilization (per-core average)
- GPU utilization (via nvidia-smi or similar)
- Order latency (per station)
- Throughput (orders/second)
- Queue depths
- Error rates

**Storage**:
- Sliding window: 30 seconds (configurable)
- Aggregations: Mean, P50, P95, P99
- Per-station and system-wide metrics

**Sampling**:
- Interval: 1 second
- Thread-based collection
- Lock-free reads for low overhead

---

### 6. Scaling Policy (`scaling_policy.py`)
**Purpose**: Determine when and how to scale workers

**Policy Types**:
- **Conservative**: Slower reactions, prevents over-scaling
- **Aggressive**: Faster reactions, optimizes for throughput

**Thresholds**:
```python
Scale Up:
  - GPU > 85% OR
  - CPU > 80% OR
  - Latency > 5s

Scale Down:
  - GPU < 40% AND
  - CPU < 30% AND
  - Latency < 2s
```

**Hysteresis**: 30-second cooldown prevents rapid scaling oscillation

---

## Data Flow

### Order Processing Flow (7 Phases)

```
1. Video Ingestion
   ├─ RTSP stream → GStreamer
   ├─ Frame extraction (~1 FPS)
   └─ OCR for order ID detection

2. Frame Storage
   ├─ Upload to MinIO (bucket: frames/)
   ├─ Path: {station_id}/{order_id}/frame_{timestamp}.jpg
   └─ EOS marker: {station_id}/{order_id}/__EOS__

3. Frame Selection
   ├─ Load all frames from MinIO
   ├─ YOLO object detection scoring
   └─ Select top-3 frames

4. VLM Request
   ├─ Create VLMRequest (station_id, order_id, frames)
   ├─ Send to vlm_request_queue
   └─ VLM Scheduler batches requests

5. VLM Inference
   ├─ OVMS processes batch
   ├─ Qwen2.5-VL-7B model
   └─ Returns detected item list

6. VLM Response
   ├─ VLM Scheduler routes response
   ├─ Station worker receives from response_queue
   └─ Parse detected items

7. Validation
   ├─ Load expected items from order manifest
   ├─ Semantic comparison service (AI matching)
   ├─ Calculate accuracy metrics
   └─ Record results and latency
```

---

## Configuration

### System Config (`config/system_config.yaml`)

```yaml
# RTSP camera streams
rtsp_urls:
  - rtsp://192.168.1.100:8554/station_1
  - rtsp://192.168.1.100:8554/station_2

# Model paths
yolo_model_path: ./model/yolo11n_openvino_model

# Data paths
inventory_path: ./config/inventory.json
orders_path: ./config/orders.json

# VLM configuration
vlm:
  ovms_url: http://ovms-vlm:8000
  model_name: vlm
  batch_window_ms: 100
  max_batch_size: 16
  max_workers: 4

# Storage configuration
storage:
  minio_endpoint: minio:9000
  minio_bucket: orders
  frames_bucket: frames
  minio_access_key: minioadmin
  minio_secret_key: minioadmin

# Scaling configuration
scaling:
  enabled: true
  min_stations: 1
  max_stations: 8
  scale_up_gpu_threshold: 85.0
  scale_down_gpu_threshold: 40.0
  hysteresis_window: 30.0
```

---

## Deployment

### Docker Compose Services

```yaml
services:
  parallel-pipeline:
    build: ./parallel-pipeline
    image: order-accuracy-parallel:latest
    container_name: oa_parallel_pipeline
    depends_on:
      - minio
    networks:
      - oa-network
    environment:
      - VLM_BACKEND=ovms
      - OVMS_URL=http://ovms-vlm:8000
      - USE_SEMANTIC_SERVICE=true
      - SEMANTIC_SERVICE_ENDPOINT=http://semantic-service:8080
    volumes:
      - ./storage/videos:/videos
      - ./datasets:/app/datasets
      - ./config:/app/config
      - ./application-service/app:/app/application-service/app
      - ./frame-selector-service/app:/app/frame-selector-service/app
```

### Running the System

```bash
# Start all services
docker compose --profile parallel up -d

# Check logs
docker logs -f oa_parallel_pipeline

# Monitor metrics
docker exec oa_parallel_pipeline python -c "from metrics_collector import MetricsStore; m=MetricsStore(); print(m.get_stats())"

# Scale manually (if autoscaling disabled)
docker exec oa_parallel_pipeline python -m main --mode fixed --stations 4
```

---

## Performance Characteristics

### Throughput
- **Single station**: ~1 order every 10-15 seconds
- **4 stations**: ~16-24 orders per minute
- **8 stations**: ~32-48 orders per minute

### Latency
- **Video processing**: 5-8 seconds
- **Frame selection**: 0.5-1 second
- **VLM inference**: 2-4 seconds (batched)
- **Validation**: 0.2-0.5 seconds
- **Total**: 8-14 seconds per order

### Resource Usage
- **CPU**: 20-40% per station (GStreamer)
- **GPU**: 60-80% with 4 stations (VLM inference)
- **Memory**: ~2 GB per station
- **Network**: ~5 Mbps per RTSP stream

---

## Error Handling

### Station Worker Failures
- Process isolation prevents cascade failures
- Metrics collector detects unresponsive workers
- Station manager restarts failed workers
- Partial results saved to MinIO

### VLM Inference Failures
- Timeout: 30 seconds per request
- Retry: 3 attempts with exponential backoff
- Fallback: Mark order as "needs review"
- Error response sent back to station

### Storage Failures
- MinIO connection retry with backoff
- Local frame buffering (in-memory)
- Graceful degradation if storage unavailable

---

## Code Statistics

- **Total Lines**: 4,292 lines of Python
- **Core Modules**: 11 files
- **Test Coverage**: Integration tests for key flows
- **Docker Image**: Intel DL Streamer 2025.2.0 base

---

## Integration Points

### Existing Services
1. **application-service**: Pipeline runner, validation logic
2. **frame-selector-service**: YOLO-based frame scoring
3. **ovms-service**: VLM model serving (Qwen2.5-VL-7B)
4. **semantic-service**: AI-powered item matching

### External Dependencies
- **GStreamer 1.0**: Video processing
- **OpenVINO 2025.4**: Model inference
- **MinIO**: Object storage
- **Docker**: Containerization

---

## Monitoring and Observability

### Key Metrics
- `orders_per_second`: System throughput
- `avg_latency_seconds`: End-to-end order latency
- `gpu_utilization_percent`: GPU usage
- `active_stations`: Current worker count
- `vlm_batch_size`: Average inference batch size

### Logging
- Structured logs with station_id context
- Log levels: DEBUG, INFO, WARNING, ERROR
- Centralized via Docker logging driver

### Health Checks
- Station worker heartbeats
- VLM scheduler responsiveness
- Storage connectivity
- Semantic service availability

---

## Future Enhancements

1. **Redis Queue Backend**: For distributed deployment
2. **Prometheus Metrics**: Export to monitoring system
3. **Grafana Dashboards**: Real-time visualization
4. **Load Testing**: Benchmark suite for stress testing
5. **A/B Testing**: Compare scaling policies
6. **Model Serving**: Multiple VLM models in parallel

---

## License

Intel Proprietary - Panther Lake Order Accuracy System
