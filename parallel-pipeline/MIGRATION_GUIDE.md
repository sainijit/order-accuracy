# Migration Guide: Per-Order → Persistent Pipeline

## Quick Start

```bash
cd /home/intel/jsaini/order-accuracy/parallel-pipeline

# 1. Clear old MinIO data (old path structure incompatible)
docker compose exec minio mc rm -r --force /data/frames/ || true

# 2. Rebuild services (ensure latest code)
docker compose build

# 3. Start with persistent pipeline
docker compose up -d

# 4. Monitor startup
docker logs -f oa_parallel_pipeline

# Expected logs:
# [station_1] Starting persistent GStreamer pipeline...
# [station_1] Persistent pipeline started (PID: XXX)
# [station_1] Frame monitor thread started
# [station_1] Health monitor thread started
# [station_1] All components started, entering main loop
```

## What Changed?

### Architecture
- **Before**: New GStreamer pipeline for every order (5-10s startup)
- **After**: One persistent pipeline per worker (starts once, runs forever)

### MinIO Structure
```diff
- frames/ORDER_123/1.jpg
+ frames/station_1/ORDER_123/frame_1707393600000_1.jpg
```

### Process Model
```diff
- Worker → per-order pipeline creation
+ Worker → persistent subprocess + monitoring threads
```

## Expected Performance

| Metric | Before | After | Gain |
|--------|--------|-------|------|
| Order Latency | 13-24s | ~6s | 75% ↓ |
| Throughput | 8/min | 18/min | 125% ↑ |
| Pipeline Starts | Every order | Once/worker | - |
| Scale-up Time | 10-15s | 2-4s | 80% ↓ |

## Verification Steps

### 1. Check Pipeline Started
```bash
docker exec oa_parallel_pipeline ps aux | grep gst-launch

# Should show ONE gst-launch process per worker
# Not restarting per order
```

### 2. Monitor Order Processing
```bash
docker logs -f oa_parallel_pipeline 2>&1 | grep "completed"

# Look for:
# Order ORDER_XXX completed: latency=6.2s (NO pipeline startup overhead!)
```

### 3. Verify MinIO Structure
```bash
docker exec oa_parallel_pipeline mc ls minio/frames/

# Should show:
# station_1/
# station_2/
# ...

docker exec oa_parallel_pipeline mc ls minio/frames/station_1/

# Should show:
# ORDER_123/
# ORDER_124/
# ...
```

### 4. Test Health Monitoring
```bash
# Kill pipeline subprocess
docker exec oa_parallel_pipeline pkill -9 gst-launch

# Watch logs - should auto-restart
docker logs -f oa_parallel_pipeline 2>&1 | grep "Restarting"

# Expected:
# [station_1] Pipeline subprocess died (exit code: -9)
# [station_1] Restarting pipeline in 2s (attempt 1)
# [station_1] Persistent pipeline started (PID: XXX)
```

## Rollback Plan

If persistent pipeline causes issues:

```bash
# Stop current deployment
docker compose down

# Checkout previous commit (before persistent pipeline)
git stash
git checkout <previous_commit>

# Rebuild and restart
docker compose build
docker compose up -d
```

## Known Issues & Solutions

### Issue: Pipeline fails to start
**Symptom**: No gst-launch process, logs show startup errors  
**Solution**: Check RTSP URL accessibility, verify GStreamer installed

### Issue: Orders not detected
**Symptom**: Pipeline running but no orders processed  
**Solution**: Check OCR is working, verify STATION_ID environment variable set

### Issue: MinIO errors
**Symptom**: "Bucket not found" or permission errors  
**Solution**: Ensure MinIO running, check credentials in docker-compose.yaml

### Issue: High memory usage
**Symptom**: Workers consuming excessive memory  
**Solution**: Check for frame leaks, verify cleanup in _load_order_frames_from_minio()

## Configuration Changes

### docker-compose.yaml
No changes required - persistent pipeline uses same configuration

### Environment Variables
New variable set automatically by worker:
```yaml
STATION_ID: <station_id>  # Set by worker for gvapython
```

### MinIO Buckets
Same buckets used:
- `frames`: Frame storage (new hierarchical structure)
- `selected`: Selected frames (unchanged)

## Performance Tuning

### Adjust Order Timeout
In `frame_pipeline.py`:
```python
_order_timeout = 10  # seconds

# Tune based on:
# - Average order duration
# - Frame rate (1 FPS = 1 frame/sec)
# - Expected frames per order
```

### Adjust Health Check Interval
In `station_worker.py`:
```python
time.sleep(5)  # Check every 5 seconds

# Tune based on:
# - Pipeline stability
# - Restart overhead
# - Monitoring resource cost
```

### Adjust Restart Backoff
In `station_worker.py`:
```python
delay = min(2 ** self._pipeline_restart_count, 60)

# Current: 2s, 4s, 8s, 16s, 32s, 60s
# Adjust base or max based on restart patterns
```

## Monitoring Dashboard

Key metrics to track:

### Latency
```bash
docker logs oa_parallel_pipeline 2>&1 | grep "latency=" | tail -20

# Target: ~6s (down from 24s)
```

### Pipeline Uptime
```bash
docker exec oa_parallel_pipeline ps -eo pid,etime,cmd | grep gst-launch

# Should show: long uptime (hours/days), not minutes
```

### Order Throughput
```bash
docker logs oa_parallel_pipeline 2>&1 | grep "completed" | wc -l

# Count orders per minute
# Target: 18/min (up from 8/min)
```

### Restart Events
```bash
docker logs oa_parallel_pipeline 2>&1 | grep "Restarting pipeline"

# Should be rare (healthy system)
# Frequent restarts indicate stability issue
```

## FAQ

**Q: Can I mix per-order and persistent pipelines?**  
A: No, all workers now use persistent pipelines. Per-order mode removed.

**Q: How many pipelines run per worker?**  
A: One persistent GStreamer subprocess per worker process.

**Q: What happens if pipeline crashes?**  
A: Health monitor detects crash, restarts with exponential backoff (up to 5 attempts).

**Q: Do I need to change order data format?**  
A: No, order manifests unchanged. Only MinIO frame storage structure changed.

**Q: Can I scale workers dynamically?**  
A: Yes! New workers start their own persistent pipeline in 2-4s and begin processing.

**Q: What about RTSP connection limits?**  
A: Each worker maintains one persistent RTSP connection. Total = number of workers.

**Q: How to debug pipeline issues?**  
A: Check GStreamer subprocess logs:
```bash
docker exec oa_parallel_pipeline cat /proc/<gst_pid>/fd/2
```

## Success Checklist

Before declaring migration complete:

- [ ] Pipeline starts once per worker (not per order)
- [ ] Order latency reduced to ~6 seconds
- [ ] No pipeline restart spam in logs
- [ ] MinIO has hierarchical structure (station_id/order_id/)
- [ ] Health monitoring working (test by killing pipeline)
- [ ] Scaling works (add worker, becomes productive in 2-4s)
- [ ] Orders processed correctly (accuracy unchanged)
- [ ] No type errors or runtime exceptions

## Support

If issues persist:
1. Check logs: `docker logs oa_parallel_pipeline`
2. Verify processes: `docker exec oa_parallel_pipeline ps aux`
3. Check MinIO: `docker exec oa_parallel_pipeline mc ls minio/frames/`
4. Review this guide's "Known Issues" section
5. Rollback if necessary (see Rollback Plan above)
