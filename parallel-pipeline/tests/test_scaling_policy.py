"""
Unit Tests for Scaling Policy

Tests scaling decision logic under various conditions.
"""

import unittest
import time
from scaling_policy import (
    ScalingPolicy,
    ScalingDecision,
    ScalingThresholds,
    ConservativeScalingPolicy,
    AggressiveScalingPolicy
)


class TestScalingPolicy(unittest.TestCase):
    """Test scaling policy logic"""
    
    def setUp(self):
        """Setup test policy"""
        self.policy = ScalingPolicy()
        # Reset hysteresis for each test
        self.policy.reset_hysteresis()
    
    def test_scale_up_conditions(self):
        """Test scale up when all conditions met"""
        decision = self.policy.evaluate(
            current_stations=2,
            cpu_utilization=60.0,  # < 80%
            gpu_utilization=70.0,  # < 85%
            avg_latency=3.5  # < 5s
        )
        
        self.assertEqual(decision.action, ScalingDecision.SCALE_UP)
        self.assertEqual(decision.target_stations, 3)
    
    def test_scale_down_gpu_overload(self):
        """Test scale down on GPU overload"""
        decision = self.policy.evaluate(
            current_stations=4,
            cpu_utilization=70.0,
            gpu_utilization=96.0,  # > 95%
            avg_latency=4.0
        )
        
        self.assertEqual(decision.action, ScalingDecision.SCALE_DOWN)
        self.assertEqual(decision.target_stations, 3)
    
    def test_scale_down_cpu_overload(self):
        """Test scale down on CPU overload"""
        decision = self.policy.evaluate(
            current_stations=4,
            cpu_utilization=92.0,  # > 90%
            gpu_utilization=85.0,
            avg_latency=4.0
        )
        
        self.assertEqual(decision.action, ScalingDecision.SCALE_DOWN)
        self.assertEqual(decision.target_stations, 3)
    
    def test_scale_down_latency_violation(self):
        """Test scale down on latency violation"""
        decision = self.policy.evaluate(
            current_stations=4,
            cpu_utilization=70.0,
            gpu_utilization=85.0,
            avg_latency=5.5  # > 5s
        )
        
        self.assertEqual(decision.action, ScalingDecision.SCALE_DOWN)
        self.assertEqual(decision.target_stations, 3)
    
    def test_no_scale_stable_state(self):
        """Test no scaling when system is stable"""
        decision = self.policy.evaluate(
            current_stations=3,
            cpu_utilization=75.0,
            gpu_utilization=82.0,
            avg_latency=4.2
        )
        
        self.assertEqual(decision.action, ScalingDecision.NONE)
        self.assertEqual(decision.target_stations, 3)
    
    def test_hysteresis_prevents_scaling(self):
        """Test hysteresis prevents rapid scaling"""
        # First scaling action
        decision1 = self.policy.evaluate(
            current_stations=2,
            cpu_utilization=60.0,
            gpu_utilization=70.0,
            avg_latency=3.5
        )
        self.assertEqual(decision1.action, ScalingDecision.SCALE_UP)
        
        # Immediate second attempt should be blocked
        decision2 = self.policy.evaluate(
            current_stations=3,
            cpu_utilization=60.0,
            gpu_utilization=70.0,
            avg_latency=3.5
        )
        self.assertEqual(decision2.action, ScalingDecision.NONE)
        self.assertIn("Hysteresis", decision2.reason)
    
    def test_min_stations_limit(self):
        """Test cannot scale below min stations"""
        decision = self.policy.evaluate(
            current_stations=1,  # At minimum
            cpu_utilization=95.0,
            gpu_utilization=98.0,
            avg_latency=6.0
        )
        
        # Should try to scale down but be blocked by limit
        self.assertEqual(decision.action, ScalingDecision.NONE)
    
    def test_max_stations_limit(self):
        """Test cannot scale above max stations"""
        decision = self.policy.evaluate(
            current_stations=8,  # At maximum
            cpu_utilization=60.0,
            gpu_utilization=70.0,
            avg_latency=3.5
        )
        
        # Should want to scale up but be blocked by limit
        self.assertEqual(decision.action, ScalingDecision.NONE)
    
    def test_conservative_policy(self):
        """Test conservative policy is more cautious"""
        policy = ConservativeScalingPolicy()
        policy.reset_hysteresis()
        
        # At 75% GPU, default policy would scale up,
        # but conservative should not
        decision = policy.evaluate(
            current_stations=2,
            cpu_utilization=65.0,
            gpu_utilization=75.0,  # > 70% (conservative threshold)
            avg_latency=3.5
        )
        
        self.assertEqual(decision.action, ScalingDecision.NONE)
    
    def test_aggressive_policy(self):
        """Test aggressive policy scales more aggressively"""
        policy = AggressiveScalingPolicy()
        policy.reset_hysteresis()
        
        # At 88% GPU, default would not scale up,
        # but aggressive should
        decision = policy.evaluate(
            current_stations=2,
            cpu_utilization=75.0,
            gpu_utilization=88.0,  # < 90% (aggressive threshold)
            avg_latency=4.0
        )
        
        self.assertEqual(decision.action, ScalingDecision.SCALE_UP)


class TestScalingThresholds(unittest.TestCase):
    """Test custom threshold configuration"""
    
    def test_custom_thresholds(self):
        """Test policy with custom thresholds"""
        thresholds = ScalingThresholds(
            scale_up_gpu_threshold=75.0,
            scale_up_cpu_threshold=75.0,
            scale_up_latency_threshold=4.0,
            hysteresis_window=10.0,
            max_stations=6
        )
        
        policy = ScalingPolicy(thresholds)
        policy.reset_hysteresis()
        
        decision = policy.evaluate(
            current_stations=2,
            cpu_utilization=70.0,
            gpu_utilization=70.0,
            avg_latency=3.5
        )
        
        self.assertEqual(decision.action, ScalingDecision.SCALE_UP)


if __name__ == '__main__':
    unittest.main()
