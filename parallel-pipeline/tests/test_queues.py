"""
Integration Tests for Queue System

Tests inter-process communication via shared queues.
"""

import unittest
import time
import multiprocessing as mp
from shared_queue import (
    QueueManager,
    QueueBackend,
    VLMRequest,
    VLMResponse,
    SharedQueue
)


class TestSharedQueue(unittest.TestCase):
    """Test shared queue operations"""
    
    def test_multiprocessing_queue(self):
        """Test multiprocessing queue backend"""
        queue = SharedQueue(
            name="test_queue",
            backend=QueueBackend.MULTIPROCESSING,
            maxsize=10
        )
        
        # Put and get
        test_data = {"key": "value", "number": 42}
        queue.put(test_data)
        
        retrieved = queue.get()
        self.assertEqual(retrieved, test_data)
        
        # Empty check
        self.assertTrue(queue.empty())
    
    def test_queue_size(self):
        """Test queue size tracking"""
        queue = SharedQueue(
            name="test_queue",
            backend=QueueBackend.MULTIPROCESSING
        )
        
        self.assertEqual(queue.qsize(), 0)
        
        queue.put({"item": 1})
        queue.put({"item": 2})
        
        self.assertEqual(queue.qsize(), 2)
        
        queue.get()
        self.assertEqual(queue.qsize(), 1)


class TestVLMRequestResponse(unittest.TestCase):
    """Test VLM request/response serialization"""
    
    def test_request_serialization(self):
        """Test VLM request can be serialized and deserialized"""
        request = VLMRequest(
            station_id="station_1",
            order_id="ORDER_123",
            frames=["frame1_data", "frame2_data"],
            timestamp=time.time(),
            priority=1
        )
        
        # Serialize
        data = request.to_dict()
        
        # Deserialize
        restored = VLMRequest.from_dict(data)
        
        self.assertEqual(restored.station_id, request.station_id)
        self.assertEqual(restored.order_id, request.order_id)
        self.assertEqual(restored.frames, request.frames)
        self.assertEqual(restored.request_id, request.request_id)
    
    def test_response_serialization(self):
        """Test VLM response can be serialized and deserialized"""
        response = VLMResponse(
            request_id="req_123",
            station_id="station_1",
            order_id="ORDER_123",
            detected_items=["burger", "fries", "coke"],
            inference_time=0.5,
            success=True
        )
        
        # Serialize
        data = response.to_dict()
        
        # Deserialize
        restored = VLMResponse.from_dict(data)
        
        self.assertEqual(restored.request_id, response.request_id)
        self.assertEqual(restored.detected_items, response.detected_items)
        self.assertEqual(restored.success, response.success)


class TestQueueManager(unittest.TestCase):
    """Test queue manager functionality"""
    
    def test_queue_manager_initialization(self):
        """Test queue manager creates queues"""
        manager = QueueManager(backend=QueueBackend.MULTIPROCESSING)
        
        self.assertIsNotNone(manager.vlm_request_queue)
        self.assertIsNotNone(manager.metrics_queue)
        self.assertIsNotNone(manager.control_queue)
    
    def test_response_queue_creation(self):
        """Test per-station response queue creation"""
        manager = QueueManager(backend=QueueBackend.MULTIPROCESSING)
        
        # Get response queue for station
        queue1 = manager.get_response_queue("station_1")
        self.assertIsNotNone(queue1)
        
        # Getting same station again should return same queue
        queue2 = manager.get_response_queue("station_1")
        self.assertIs(queue1, queue2)
        
        # Different station should get different queue
        queue3 = manager.get_response_queue("station_2")
        self.assertIsNot(queue1, queue3)


def worker_process(queue: SharedQueue, data_to_send: dict):
    """Worker process for inter-process test"""
    time.sleep(0.1)
    queue.put(data_to_send)


class TestInterProcessCommunication(unittest.TestCase):
    """Test inter-process communication"""
    
    def test_multiprocess_communication(self):
        """Test data can be passed between processes"""
        queue = SharedQueue(
            name="test_ipc",
            backend=QueueBackend.MULTIPROCESSING
        )
        
        test_data = {"message": "hello from worker", "value": 99}
        
        # Start worker process
        process = mp.Process(
            target=worker_process,
            args=(queue, test_data)
        )
        process.start()
        
        # Wait for data
        received = queue.get(timeout=2.0)
        
        # Verify
        self.assertEqual(received, test_data)
        
        # Cleanup
        process.join()


if __name__ == '__main__':
    unittest.main()
