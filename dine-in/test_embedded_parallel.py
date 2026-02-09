#!/usr/bin/env python3
"""
Test Embedded VLM backend with sequential processing using direct VLM instance access
This script directly calls vlm_instance.process() to test sequential processing behavior
Note: Embedded backend is not thread-safe, so requests are processed sequentially
"""
import time
import numpy as np
from pathlib import Path
from PIL import Image
import sys
import os
import threading

# Add application service path
APP_PATH = Path(__file__).parent.parent / 'application-service' / 'app'
sys.path.insert(0, str(APP_PATH))

# Get script directory
SCRIPT_DIR = Path(__file__).parent

# Track request processing
request_lock = threading.Lock()
processing_order = []

def load_image(image_path):
    """Load and prepare image as numpy array"""
    with Image.open(image_path) as img:
        img_rgb = img.convert('RGB').resize((512, 512))
        return np.array(img_rgb)

def process_vlm_request(image_path, request_id):
    """Process image using VLM instance directly"""
    try:
        # Import VLM instance (lazy import to ensure proper initialization)
        from vlm_service import vlm_instance
        
        # Track when this request starts processing
        with request_lock:
            start_order = len(processing_order)
            processing_order.append(f"Request {request_id}")
        
        # Load image
        img_array = load_image(image_path)
        
        # Process with VLM - this is where queuing happens
        start = time.time()
        result = vlm_instance.process([img_array])
        elapsed = time.time() - start
        
        # Debug: Print raw VLM output
        print(f"      DEBUG - Raw result: {result}")
        
        return {
            'request_id': request_id,
            'image': str(image_path),
            'success': True,
            'elapsed': elapsed,
            'detected_items': result['items'],
            'num_items': len(result['items']),
            'inference_time': result['inference_time_sec'],
            'start_order': start_order
        }
        
    except Exception as e:
        return {
            'request_id': request_id,
            'image': str(image_path),
            'success': False,
            'error': str(e)
        }

def run_parallel_test(num_concurrent=4, num_batches=2):
    """Run sequential test with embedded VLM backend"""
    
    print("="*70)
    print("Embedded VLM Backend - Sequential Processing Test")
    print("="*70)
    
    # Pre-initialize VLM instance
    print("\nInitializing VLM instance...")
    init_start = time.time()
    try:
        from vlm_service import vlm_instance, VLM_BACKEND
        print(f"✓ VLM instance loaded (backend: {VLM_BACKEND})")
        print(f"  Initialization time: {time.time() - init_start:.2f}s")
    except Exception as e:
        print(f"✗ Failed to load VLM instance: {e}")
        return []
    
    # Get available images - check multiple locations
    # When running inside container, images are in /tmp/images
    image_paths = []
    possible_dirs = [
        Path('/tmp/images'),  # Container location
        SCRIPT_DIR / 'images',  # Local script dir
        Path(__file__).parent.parent / 'images',  # Parent dir
    ]
    
    for img_dir in possible_dirs:
        if img_dir.exists():
            image_paths = [img_dir / f'plate{i}.png' for i in range(4, 8)]
            if all(p.exists() for p in image_paths):
                break
    
    if not image_paths or not all(p.exists() for p in image_paths):
        print(f"✗ Could not find test images in any of: {[str(d) for d in possible_dirs]}")
        return []
    
    images = image_paths
    
    # Test configuration
    total_requests = num_concurrent * num_batches
    print(f"\nTest Configuration:")
    print(f"  Requests per batch: {num_concurrent}")
    print(f"  Number of batches: {num_batches}")
    print(f"  Total requests: {total_requests}")
    print(f"  Images: {len(images)} plates")
    print(f"  Backend: {VLM_BACKEND.upper()}")
    print(f"  Method: Sequential vlm_instance.process() calls")
    print(f"  Note: Embedded backend is NOT thread-safe")
    
    # Run parallel requests
    all_results = []
    test_start = time.time()
    
    for batch_num in range(num_batches):
        print(f"\n{'='*70}")
        print(f"Batch {batch_num + 1}/{num_batches}")
        print('='*70)
        
        batch_start = time.time()
        processing_order.clear()
        
        # Process requests sequentially (embedded backend is not thread-safe)
        print(f"\n  Processing {num_concurrent} requests sequentially...")
        
        completed_count = 0
        for i in range(num_concurrent):
            request_id = batch_num * num_concurrent + i
            image_path = images[i % len(images)]
            print(f"  [{i+1}/{num_concurrent}] Processing request {request_id}: {image_path.name}")
            
            # Process sequentially
            result = process_vlm_request(image_path, request_id)
            all_results.append(result)
            completed_count += 1
            
            if result['success']:
                print(f"      ✓ Completed: {result['elapsed']:.2f}s - {result['num_items']} items "
                      f"(VLM: {result['inference_time']:.2f}s)")
            else:
                print(f"      ✗ Failed: {result.get('error', 'Unknown error')}")
        
        batch_elapsed = time.time() - batch_start
        print(f"\n  Batch completed in {batch_elapsed:.2f}s")
        
        # Show processing order
        if len(processing_order) > 1:
            print(f"  Processing order: {' → '.join(processing_order)}")
    
    test_elapsed = time.time() - test_start
    
    # Analyze results
    print(f"\n{'='*70}")
    print("Results Summary")
    print('='*70)
    
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    print(f"\nRequests:")
    print(f"  Total: {len(all_results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    
    if successful:
        latencies = [r['elapsed'] for r in successful]
        inference_times = [r['inference_time'] for r in successful]
        
        print(f"\nEnd-to-End Latency Statistics:")
        print(f"  Min: {min(latencies):.2f}s")
        print(f"  Max: {max(latencies):.2f}s")
        print(f"  Mean: {sum(latencies)/len(latencies):.2f}s")
        print(f"  Std Dev: {np.std(latencies):.2f}s")
        
        print(f"\nVLM Inference Time Statistics:")
        print(f"  Min: {min(inference_times):.2f}s")
        print(f"  Max: {max(inference_times):.2f}s")
        print(f"  Mean: {sum(inference_times)/len(inference_times):.2f}s")
        print(f"  Std Dev: {np.std(inference_times):.2f}s")
        
        print(f"\nThroughput:")
        print(f"  Total test time: {test_elapsed:.2f}s")
        print(f"  Throughput: {len(successful)/test_elapsed:.2f} requests/second")
        
        # Analyze concurrency behavior
        print(f"\nConcurrency Analysis:")
        avg_latency = sum(latencies) / len(latencies)
        expected_sequential_time = avg_latency * len(successful)
        actual_time = test_elapsed
        
        print(f"  Average single request time: {avg_latency:.2f}s")
        print(f"  Expected if fully sequential: {expected_sequential_time:.2f}s")
        print(f"  Actual concurrent time: {actual_time:.2f}s")
        efficiency_ratio = actual_time / expected_sequential_time
        print(f"  Efficiency ratio: {efficiency_ratio:.2f}x")
        
        if efficiency_ratio <= 1.1:
            print(f"  Result: ✓ Expected sequential processing (within 10% overhead)")
        else:
            print(f"  Result: WARNING Higher overhead than expected")
        
        # Check latency variance
        latency_variance = max(latencies) - min(latencies)
        if latency_variance > avg_latency * 0.5:
            print(f"\n  Note: High latency variance ({latency_variance:.2f}s)")
            print(f"        Later requests waited for earlier ones (queue effect)")
    
    # Show sample responses
    if successful:
        print(f"\n{'='*70}")
        print("Sample Results")
        print('='*70)
        for i, result in enumerate(successful[:2]):
            print(f"\n[Request {result['request_id']} - {Path(result['image']).name}]")
            print(f"  End-to-end time: {result['elapsed']:.2f}s")
            print(f"  VLM inference time: {result['inference_time']:.2f}s")
            print(f"  Overhead: {result['elapsed'] - result['inference_time']:.2f}s")
            print(f"  Detected items: {result['num_items']}")
            if result['detected_items']:
                for item in result['detected_items'][:3]:
                    print(f"    - {item.get('name', 'Unknown')} x {item.get('quantity', 0)}")
    
    # Comparison summary
    print(f"\n{'='*70}")
    print("Backend Comparison Summary")
    print('='*70)
    print("\n📊 OVMS Backend (with VLM_CB pipeline):")
    print("  ✓ True batching support")
    print("  ✓ Processes multiple requests simultaneously on GPU")
    print("  ✓ Better throughput under high load")
    print("  ✓ Lower latency for concurrent requests")
    print("  ⚠️  Higher memory usage (~21GB GPU memory)")
    
    print("\n📊 Embedded Backend (OpenVINO GenAI):")
    print("  ✗ No batching support")
    print("  ✗ Sequential processing (uses internal queue)")
    print("  ✗ Concurrent requests wait for each other")
    print("  ✓ Lower memory footprint (~11GB GPU memory)")
    print("  ✓ Simpler deployment (no OVMS server)")
    print("  ✓ Good for single-request scenarios")
    
    print("\n💡 Recommendation:")
    print("  - Use OVMS for production with concurrent users")
    print("  - Use Embedded for single-user or low-concurrency scenarios")
    
    return all_results

if __name__ == "__main__":
    import sys
    
    # Parse arguments
    num_concurrent = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    num_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    print("\n" + "="*70)
    print("NOTE: This test must run inside the application-service Docker container")
    print("      Run with: docker exec oa_application python3 /path/to/script.py")
    print("="*70)
    
    # Check if running in container
    if not os.path.exists('/.dockerenv'):
        print("\n⚠️  ERROR: This script must be run inside the Docker container")
        print("\nTo run this test:")
        print("  1. Copy script to container:")
        print("     docker cp test_embedded_parallel.py oa_application:/tmp/")
        print("  2. Execute inside container:")
        print("     docker exec oa_application python3 /tmp/test_embedded_parallel.py 4 1")
        sys.exit(1)
    
    results = run_parallel_test(num_concurrent, num_batches)
    
    print(f"\n{'='*70}")
    print("Test Complete!")
    print('='*70)
