#!/usr/bin/env python3
"""
Quick test script to verify parallel pipeline setup.

This script tests basic functionality without requiring full integration.
"""

import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported"""
    logger.info("Testing module imports...")
    
    try:
        from shared_queue import QueueManager, VLMRequest, VLMResponse
        from metrics_collector import MetricsCollector, MetricsStore
        from scaling_policy import ScalingPolicy, ScalingDecision
        from vlm_scheduler import VLMScheduler
        from station_worker import StationWorker
        from station_manager import StationManager
        from config import SystemConfig
        from benchmark_runner import BenchmarkRunner
        
        logger.info("✅ All modules imported successfully")
        return True
    
    except Exception as e:
        logger.error(f"❌ Import error: {e}")
        return False


def test_config():
    """Test configuration system"""
    logger.info("Testing configuration...")
    
    try:
        from config import create_default_config
        
        config = create_default_config()
        config_dict = config.to_dict()
        
        assert 'vlm' in config_dict
        assert 'scaling' in config_dict
        assert 'storage' in config_dict
        
        logger.info("✅ Configuration system working")
        return True
    
    except Exception as e:
        logger.error(f"❌ Config error: {e}")
        return False


def test_queues():
    """Test queue system"""
    logger.info("Testing queue system...")
    
    try:
        from shared_queue import QueueManager, QueueBackend, VLMRequest
        import time
        
        # Create queue manager
        manager = QueueManager(backend=QueueBackend.MULTIPROCESSING)
        
        # Test VLM request queue
        request = VLMRequest(
            station_id="test_station",
            order_id="TEST_ORDER",
            frames=["frame1", "frame2"],
            timestamp=time.time()
        )
        
        manager.vlm_request_queue.put(request.to_dict())
        retrieved = manager.vlm_request_queue.get()
        
        assert retrieved is not None
        assert retrieved['order_id'] == "TEST_ORDER"
        
        logger.info("✅ Queue system working")
        return True
    
    except Exception as e:
        logger.error(f"❌ Queue error: {e}")
        return False


def test_metrics():
    """Test metrics collection"""
    logger.info("Testing metrics system...")
    
    try:
        from metrics_collector import MetricsStore
        
        store = MetricsStore(window_size=10)
        
        # Record some metrics
        store.record_cpu(75.0)
        store.record_gpu(80.0)
        store.record_latency("station_1", 4.5)
        
        # Retrieve metrics
        cpu_avg = store.get_cpu_avg()
        gpu_avg = store.get_gpu_avg()
        latency_avg = store.get_latency_avg("station_1")
        
        assert cpu_avg == 75.0
        assert gpu_avg == 80.0
        assert latency_avg == 4.5
        
        logger.info("✅ Metrics system working")
        return True
    
    except Exception as e:
        logger.error(f"❌ Metrics error: {e}")
        return False


def test_scaling_policy():
    """Test scaling policy logic"""
    logger.info("Testing scaling policy...")
    
    try:
        from scaling_policy import ScalingPolicy, ScalingDecision
        
        policy = ScalingPolicy()
        policy.reset_hysteresis()
        
        # Test scale up
        decision = policy.evaluate(
            current_stations=2,
            cpu_utilization=60.0,
            gpu_utilization=70.0,
            avg_latency=3.5
        )
        
        assert decision.action == ScalingDecision.SCALE_UP
        assert decision.target_stations == 3
        
        # Test scale down
        policy.reset_hysteresis()
        decision = policy.evaluate(
            current_stations=4,
            cpu_utilization=95.0,
            gpu_utilization=98.0,
            avg_latency=6.0
        )
        
        assert decision.action == ScalingDecision.SCALE_DOWN
        assert decision.target_stations == 3
        
        logger.info("✅ Scaling policy working")
        return True
    
    except Exception as e:
        logger.error(f"❌ Scaling policy error: {e}")
        return False


def test_config_files():
    """Test configuration file creation"""
    logger.info("Testing configuration files...")
    
    try:
        from config import create_default_config
        
        config = create_default_config()
        
        # Save to temp location
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.save_yaml(f.name)
            logger.info(f"  Created test config: {f.name}")
        
        logger.info("✅ Configuration files working")
        return True
    
    except Exception as e:
        logger.error(f"❌ Config file error: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Parallel Pipeline - System Verification")
    logger.info("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_config),
        ("Queue System", test_queues),
        ("Metrics Collection", test_metrics),
        ("Scaling Policy", test_scaling_policy),
        ("Config Files", test_config_files)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        passed = test_func()
        results.append((test_name, passed))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for test_name, passed_flag in results:
        status = "✅ PASS" if passed_flag else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
    
    logger.info(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! System is ready.")
        logger.info("\nNext steps:")
        logger.info("1. Review QUICKSTART.md for usage")
        logger.info("2. Edit config/system_config.yaml")
        logger.info("3. Run: python main.py fixed --stations 1")
        return 0
    else:
        logger.error("\n⚠️  Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
