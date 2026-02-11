# GPU Enablement Guide for Order Accuracy System

## Current Status
- **System**: Intel Xeon Platinum 8468V with 2x Intel Data Center GPUs (Device ID: 0x2710)
- **GPU Status**: DISABLED in BIOS/firmware
- **Current Configuration**: Running on CPU (fully functional)
- **OVMS Model**: Qwen/Qwen2.5-VL-7B-Instruct (AVAILABLE on CPU)

## Detected Hardware
```
6d:00.0 Co-processor [0b40]: Intel Corporation Device [8086:2710]
ea:00.0 Co-processor [0b40]: Intel Corporation Device [8086:2710]
```

Device 0x2710 = Intel Data Center GPU (likely Max Series or Flex Series)

## Current GPU Status
- Memory regions: **DISABLED**
- No kernel driver loaded
- Device control: `I/O- Mem- BusMaster-`

## Steps to Enable GPU Acceleration

### 1. Enable GPUs in BIOS
- Access BIOS/UEFI settings
- Navigate to PCIe Configuration or Advanced settings
- Enable PCIe device at bus 6d:00.0 and ea:00.0
- Save and reboot

### 2. Install Intel GPU Drivers on Host

```bash
# Add Intel GPU package repository
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg

echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu.list

sudo apt update

# Install compute runtime and Level Zero
sudo apt install -y \
  intel-opencl-icd \
  intel-level-zero-gpu \
  level-zero \
  intel-media-va-driver-non-free \
  libmfx1 \
  libmfxgen1 \
  libvpl2

# Install development tools (optional)
sudo apt install -y \
  clinfo \
  vainfo \
  intel-gpu-tools
```

### 3. Verify GPU Detection

```bash
# Check if render nodes are created
ls -la /dev/dri/

# Should see renderD128, renderD129, etc.

# Check OpenCL devices
clinfo

# Check Level Zero devices  
sudo /usr/local/bin/level_zero_loader --list
```

### 4. Update graph.pbtxt for GPU

```bash
cd /home/intel/sainijit/order-accuracy/models/Qwen/Qwen2.5-VL-7B-Instruct

# Restore GPU configuration
cp graph.pbtxt.gpu_backup graph.pbtxt

# Or manually change:
sed -i 's/device: "CPU"/device: "GPU"/' graph.pbtxt
```

### 5. Update OVMS config.json for GPU

Edit `/home/intel/sainijit/order-accuracy/models/config.json`:
```json
{
    "model_config_list": [
        {
            "config": {
                "name": "Qwen/Qwen2.5-VL-7B-Instruct",
                "base_path": "Qwen/Qwen2.5-VL-7B-Instruct",
                "target_device": "GPU",
                "plugin_config": {
                    "NUM_STREAMS": "1",
                    "PERFORMANCE_HINT": "LATENCY",
                    "CACHE_DIR": "/tmp/ov_cache"
                }
            }
        }
    ]
}
```

### 6. Update Docker Compose for GPU Access

Ensure proper device mounting in `docker-compose.yaml`:
```yaml
ovms-vlm:
  devices:
    - /dev/dri:/dev/dri
  group_add:
    - "44"   # video group
    - "992"  # render group (check with: getent group render)
```

### 7. Restart OVMS

```bash
cd /home/intel/sainijit/order-accuracy
docker compose --profile ovms restart ovms-vlm

# Wait 30-60 seconds for model loading
sleep 40

# Check status
curl http://localhost:8001/v1/config
```

### 8. Verify GPU Usage

```bash
# Check OVMS logs for GPU initialization
docker logs oa_ovms_vlm 2>&1 | grep -iE "gpu|device|available"

# Should see:
# "Available devices for Open VINO: CPU GPU.0 GPU.1"

# Monitor GPU utilization
sudo intel_gpu_top
```

## Performance Expectations

### Current (CPU):
- Xeon Platinum 8468V: 48 cores, excellent for inference
- Expected throughput: 10-30 tokens/sec per request
- Memory: Uses system RAM (~8GB for model)

### With GPU:
- Intel Data Center GPU: Optimized for AI workloads
- Expected throughput: 50-200+ tokens/sec per request
- Memory: Uses GPU VRAM (faster than system RAM)
- Better concurrency: Handle multiple requests simultaneously

## Troubleshooting

### If GPUs still show as disabled after BIOS change:
```bash
# Rescan PCI bus
echo 1 | sudo tee /sys/bus/pci/rescan

# Check if device is enabled
lspci -vvv -s 6d:00.0 | grep -iE "control|memory"
```

### If driver installation fails:
- Ensure kernel version >= 5.15
- Check for conflicting drivers: `lsmod | grep -i gpu`

### If OVMS still uses CPU:
- Check graph.pbtxt: `cat models/Qwen/*/graph.pbtxt | grep device`
- Check config.json: `cat models/config.json | grep target_device`
- Verify container can access GPU: `docker exec oa_ovms_vlm ls -la /dev/dri`

## Current Working Configuration (CPU)

The system is **fully functional on CPU**. No changes needed for basic operation.

Files configured for CPU:
- `/home/intel/sainijit/order-accuracy/models/config.json` - target_device: CPU
- `/home/intel/sainijit/order-accuracy/models/Qwen/Qwen2.5-VL-7B-Instruct/graph.pbtxt` - device: "CPU"

Backup file for GPU configuration:
- `/home/intel/sainijit/order-accuracy/models/Qwen/Qwen2.5-VL-7B-Instruct/graph.pbtxt.gpu_backup`

## References

- [OpenVINO VLM Demo](https://docs.openvino.ai/2025/model-server/ovms_demos_continuous_batching_vlm.html)
- [Intel GPU Drivers](https://dgpu-docs.intel.com/driver/installation.html)
- [OpenVINO GPU Plugin](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html)
