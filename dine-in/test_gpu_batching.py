#!/usr/bin/env python3
"""
Test OVMS VLM with concurrent requests to enable batching
Monitor GPU utilization during the test
"""
import requests
import base64
import json
import time
import subprocess
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Get script directory to resolve relative paths
SCRIPT_DIR = Path(__file__).parent

def get_gpu_stats():
    """Get GPU stats using docker stats (non-blocking)"""
    try:
        result = subprocess.run(
            ['docker', 'stats', '--no-stream', '--format', '{{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}', '28d5be0992e6'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return None

def send_vlm_request(image_path, request_id):
    """Send a single VLM request"""
    try:
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Create request
        request_data = {
            "model": "Qwen/Qwen2.5-VL-7B-Instruct-ov-int8",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "List all food items visible on this plate:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]
            }],
            "max_completion_tokens": 100
        }
        
        start = time.time()
        response = requests.post(
            "http://localhost:8001/v3/chat/completions",
            headers={"Content-Type": "application/json"},
            json=request_data,
            timeout=180
        )
        elapsed = time.time() - start
        
        result = {
            'request_id': request_id,
            'image': image_path,
            'status': response.status_code,
            'elapsed': elapsed,
            'success': response.status_code == 200
        }
        
        if response.status_code == 200:
            data = response.json()
            result['content'] = data['choices'][0]['message']['content']
            result['tokens'] = data['usage']['total_tokens']
        else:
            result['error'] = response.text
            
        return result
        
    except Exception as e:
        return {
            'request_id': request_id,
            'image': image_path,
            'success': False,
            'error': str(e)
        }

def run_batch_test(num_concurrent=4, num_batches=2):
    """Run batching test"""
    
    print("="*70)
    print("OVMS VLM Batching Test with GPU Monitoring")
    print("="*70)
    
    # Get available images (using absolute paths)
    images = [str(SCRIPT_DIR / 'images' / f'plate{i}.png') for i in range(4, 8)]
    
    # Prepare test configuration
    total_requests = num_concurrent * num_batches
    print(f"\nTest Configuration:")
    print(f"  Concurrent requests per batch: {num_concurrent}")
    print(f"  Number of batches: {num_batches}")
    print(f"  Total requests: {total_requests}")
    print(f"  Images: {len(images)} plates")
    
    # Get baseline
    print(f"\nBaseline Docker stats:")
    baseline = get_gpu_stats()
    if baseline:
        print(f"  {baseline}")
    
    # Run batched requests
    all_results = []
    test_start = time.time()
    
    for batch_num in range(num_batches):
        print(f"\n{'='*70}")
        print(f"Batch {batch_num + 1}/{num_batches}")
        print('='*70)
        
        batch_start = time.time()
        
        # Submit concurrent requests
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = []
            for i in range(num_concurrent):
                request_id = batch_num * num_concurrent + i
                image_path = images[i % len(images)]
                future = executor.submit(send_vlm_request, image_path, request_id)
                futures.append(future)
                print(f"  Submitted request {request_id}: {image_path}")
            
            # Collect results
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                
                if result['success']:
                    print(f"  ✓ Request {result['request_id']}: {result['elapsed']:.2f}s - {result['tokens']} tokens")
                else:
                    print(f"  ✗ Request {result['request_id']}: Failed - {result.get('error', 'Unknown error')}")
        
        batch_elapsed = time.time() - batch_start
        print(f"\n  Batch completed in {batch_elapsed:.2f}s")
        
        # Get stats after batch
        stats = get_gpu_stats()
        if stats:
            print(f"  Docker stats: {stats}")
    
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
        print(f"\nLatency Statistics:")
        print(f"  Min: {min(latencies):.2f}s")
        print(f"  Max: {max(latencies):.2f}s")
        print(f"  Mean: {sum(latencies)/len(latencies):.2f}s")
        print(f"  Total test time: {test_elapsed:.2f}s")
        print(f"  Throughput: {len(successful)/test_elapsed:.2f} requests/second")
        
        # Calculate batching efficiency
        single_request_time = 4.7  # From previous single request test
        expected_sequential_time = len(successful) * single_request_time
        speedup = expected_sequential_time / test_elapsed
        print(f"\n  Expected sequential time: {expected_sequential_time:.2f}s")
        print(f"  Actual batched time: {test_elapsed:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
    
    # Show sample responses
    if successful:
        print(f"\n{'='*70}")
        print("Sample Responses")
        print('='*70)
        for i, result in enumerate(successful[:2]):
            print(f"\n[Request {result['request_id']} - {result['image']}]")
            print(f"Time: {result['elapsed']:.2f}s")
            print(f"Response: {result['content'][:100]}...")
    
    return all_results

if __name__ == "__main__":
    import sys
    
    # Parse arguments
    num_concurrent = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    num_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    
    results = run_batch_test(num_concurrent, num_batches)
    
    print(f"\n{'='*70}")
    print("Test Complete!")
    print('='*70)
