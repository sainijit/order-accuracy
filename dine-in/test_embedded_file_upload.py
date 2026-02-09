#!/usr/bin/env python3
"""
Test Embedded VLM backend with concurrent file uploads
Uses the new /process_image endpoint for direct image processing
"""
import requests
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Get script directory to resolve relative paths
SCRIPT_DIR = Path(__file__).parent

def send_image_file(image_path, request_id):
    """Send image file directly to embedded backend"""
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/png')}
            
            start = time.time()
            response = requests.post(
                "http://localhost:8000/process_image",
                files=files,
                timeout=180
            )
            elapsed = time.time() - start
        
        result = {
            'request_id': request_id,
            'image': str(image_path),
            'status': response.status_code,
            'elapsed': elapsed,
            'success': response.status_code == 200
        }
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                result['detected_items'] = data.get('detected_items', [])
                result['num_items'] = len(result['detected_items'])
                result['inference_time'] = data.get('inference_time_sec', 0)
                result['total_time'] = data.get('total_time_sec', 0)
            else:
                result['success'] = False
                result['error'] = f"Processing failed: {data.get('reason', 'Unknown')}"
        else:
            result['error'] = response.text
            
        return result
        
    except Exception as e:
        return {
            'request_id': request_id,
            'image': str(image_path),
            'success': False,
            'error': str(e)
        }

def run_batch_test(num_concurrent=4, num_batches=2):
    """Run concurrent test for embedded backend with file uploads"""
    
    print("="*70)
    print("Embedded VLM Backend - File Upload Concurrent Test")
    print("="*70)
    
    # Get available images (using absolute paths)
    images = [SCRIPT_DIR / 'images' / f'plate{i}.png' for i in range(4, 8)]
    
    # Prepare test configuration
    total_requests = num_concurrent * num_batches
    print(f"\nTest Configuration:")
    print(f"  Concurrent requests per batch: {num_concurrent}")
    print(f"  Number of batches: {num_batches}")
    print(f"  Total requests: {total_requests}")
    print(f"  Images: {len(images)} plates")
    print(f"  Backend: Embedded OpenVINO GenAI")
    print(f"  Method: Direct file upload to /process_image endpoint")
    
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
                future = executor.submit(send_image_file, image_path, request_id)
                futures.append(future)
                print(f"  Submitted request {request_id}: {image_path.name}")
            
            # Collect results
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                
                if result['success']:
                    print(f"  ✓ Request {result['request_id']}: {result['elapsed']:.2f}s - {result['num_items']} items (VLM: {result['inference_time']:.2f}s)")
                else:
                    print(f"  ✗ Request {result['request_id']}: Failed - {result.get('error', 'Unknown error')}")
        
        batch_elapsed = time.time() - batch_start
        print(f"\n  Batch completed in {batch_elapsed:.2f}s")
    
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
        
        print(f"\nVLM Inference Time Statistics:")
        print(f"  Min: {min(inference_times):.2f}s")
        print(f"  Max: {max(inference_times):.2f}s")
        print(f"  Mean: {sum(inference_times)/len(inference_times):.2f}s")
        
        print(f"\nThroughput:")
        print(f"  Total test time: {test_elapsed:.2f}s")
        print(f"  Throughput: {len(successful)/test_elapsed:.2f} requests/second")
        
        # Calculate efficiency
        print(f"\nConcurrency Behavior:")
        print(f"  Note: Embedded backend processes requests sequentially")
        print(f"        (VLM pipeline not thread-safe, uses queue)")
        print(f"        Concurrent requests are queued and processed one at a time")
        
        single_request_time = sum(latencies) / len(latencies)
        expected_concurrent_time = single_request_time  # If true parallel
        actual_vs_expected = test_elapsed / expected_concurrent_time
        print(f"  Actual time vs single request: {actual_vs_expected:.2f}x")
    
    # Show sample responses
    if successful:
        print(f"\n{'='*70}")
        print("Sample Results")
        print('='*70)
        for i, result in enumerate(successful[:2]):
            print(f"\n[Request {result['request_id']} - {Path(result['image']).name}]")
            print(f"  End-to-end time: {result['elapsed']:.2f}s")
            print(f"  VLM inference time: {result['inference_time']:.2f}s")
            print(f"  Detected items: {result['num_items']}")
            if result['detected_items']:
                for item in result['detected_items'][:3]:
                    print(f"    - {item.get('name', 'Unknown')} x {item.get('quantity', 0)}")
    
    print(f"\n{'='*70}")
    print("Backend Comparison")
    print('='*70)
    print("OVMS Backend (with VLM_CB):")
    print("  ✓ Supports true batching")
    print("  ✓ Can process multiple requests simultaneously")
    print("  ✓ Better GPU utilization with concurrent requests")
    print("\nEmbedded Backend (OpenVINO GenAI):")
    print("  ✗ No batching support")
    print("  ✗ Sequential processing only (queued)")
    print("  ✓ Lower memory footprint")
    print("  ✓ Simpler deployment")
    
    return all_results

if __name__ == "__main__":
    import sys
    
    # Parse arguments
    num_concurrent = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    num_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    results = run_batch_test(num_concurrent, num_batches)
    
    print(f"\n{'='*70}")
    print("Test Complete!")
    print('='*70)
