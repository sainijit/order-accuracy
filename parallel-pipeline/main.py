"""
Main Entry Point for Parallel Order Accuracy System

Provides simple CLI for running the system in different modes:
- Production mode with autoscaling
- Fixed-station mode
- Benchmark mode
"""

import argparse
import logging
import sys
import signal
from pathlib import Path

from station_manager import StationManager
from vlm_scheduler import VLMScheduler
from config import SystemConfig
from scaling_policy import ConservativeScalingPolicy, AggressiveScalingPolicy
from shared_queue import QueueBackend
from benchmark_runner import run_standard_benchmark


def setup_logging(level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    log_level = getattr(logging, level.upper())
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def run_production(args):
    """Run in production mode with autoscaling"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Order Accuracy System in PRODUCTION mode")
    
    # Load configuration
    config = SystemConfig.from_yaml(args.config)
    logger.info(f"Loaded configuration from: {args.config}")
    
    # Select scaling policy
    if args.scaling_policy == "conservative":
        scaling_policy = ConservativeScalingPolicy()
    elif args.scaling_policy == "aggressive":
        scaling_policy = AggressiveScalingPolicy()
    else:
        scaling_policy = None  # Use default
    
    # Queue backend
    queue_backend = (
        QueueBackend.REDIS if args.queue_backend == "redis"
        else QueueBackend.MULTIPROCESSING
    )
    
    # Create station manager
    manager = StationManager(
        config=config.to_dict(),
        initial_stations=args.initial_stations,
        scaling_policy=scaling_policy,
        queue_backend=queue_backend
    )
    
    # Create VLM scheduler
    scheduler = VLMScheduler(
        queue_manager=manager.queue_manager,
        ovms_url=config.vlm.ovms_url,
        model_name=config.vlm.model_name,
        batch_window_ms=config.vlm.batch_window_ms,
        max_batch_size=config.vlm.max_batch_size,
        max_workers=config.vlm.max_workers
    )
    
    # Start scheduler
    scheduler.start()
    logger.info("VLM Scheduler started")
    
    try:
        # Start manager (runs until interrupted)
        manager.start()
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Cleanup
        logger.info("Shutting down...")
        scheduler.stop()
        manager.stop()
        logger.info("Shutdown complete")


def run_fixed(args):
    """Run with fixed number of stations (no autoscaling)"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Order Accuracy System with FIXED {args.stations} stations")
    
    # Load configuration
    config = SystemConfig.from_yaml(args.config)
    
    # Queue backend
    queue_backend = (
        QueueBackend.REDIS if args.queue_backend == "redis"
        else QueueBackend.MULTIPROCESSING
    )
    
    # Create station manager
    manager = StationManager(
        config=config.to_dict(),
        initial_stations=0,  # Start with 0, manually set
        queue_backend=queue_backend
    )
    
    # Disable autoscaling
    manager.disable_autoscaling()
    
    # Set fixed station count
    manager.set_station_count(args.stations)
    
    # Create VLM scheduler
    scheduler = VLMScheduler(
        queue_manager=manager.queue_manager,
        ovms_url=config.vlm.ovms_url,
        model_name=config.vlm.model_name,
        batch_window_ms=config.vlm.batch_window_ms,
        max_batch_size=config.vlm.max_batch_size,
        max_workers=config.vlm.max_workers
    )
    
    # Start scheduler
    scheduler.start()
    logger.info("VLM Scheduler started")
    
    try:
        # Start manager
        manager.start()
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Cleanup
        logger.info("Shutting down...")
        scheduler.stop()
        manager.stop()
        logger.info("Shutdown complete")


def run_benchmark(args):
    """Run benchmark mode"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Order Accuracy System in BENCHMARK mode")
    
    # Load configuration
    config = SystemConfig.from_yaml(args.config)
    
    # Run standard or custom benchmark
    if args.custom_scenarios:
        logger.info("Running custom benchmark scenarios")
        # TODO: Implement custom scenario loading from file
        logger.error("Custom scenarios not yet implemented")
        sys.exit(1)
    else:
        logger.info("Running standard benchmark suite")
        results = run_standard_benchmark(
            config=config.to_dict(),
            output_dir=args.output_dir
        )
    
    logger.info(f"Benchmark complete. Results saved to: {args.output_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Parallel Order Accuracy Pipeline with Autoscaling"
    )
    
    # Common arguments
    parser.add_argument(
        '--config',
        type=str,
        default='./config/system_config.yaml',
        help='Path to configuration file (default: ./config/system_config.yaml)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Log file path (default: stdout only)'
    )
    
    parser.add_argument(
        '--queue-backend',
        type=str,
        default='multiprocessing',
        choices=['multiprocessing', 'redis'],
        help='Queue backend (default: multiprocessing)'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='mode', help='Execution mode')
    
    # Production mode
    prod_parser = subparsers.add_parser(
        'production',
        help='Run in production mode with autoscaling'
    )
    prod_parser.add_argument(
        '--initial-stations',
        type=int,
        default=1,
        help='Initial number of stations (default: 1)'
    )
    prod_parser.add_argument(
        '--scaling-policy',
        type=str,
        default='default',
        choices=['default', 'conservative', 'aggressive'],
        help='Scaling policy (default: default)'
    )
    
    # Fixed mode
    fixed_parser = subparsers.add_parser(
        'fixed',
        help='Run with fixed number of stations (no autoscaling)'
    )
    fixed_parser.add_argument(
        '--stations',
        type=int,
        required=True,
        help='Number of stations to run'
    )
    
    # Benchmark mode
    bench_parser = subparsers.add_parser(
        'benchmark',
        help='Run benchmark mode'
    )
    bench_parser.add_argument(
        '--output-dir',
        type=str,
        default='./benchmark_results',
        help='Output directory for results (default: ./benchmark_results)'
    )
    bench_parser.add_argument(
        '--custom-scenarios',
        type=str,
        default=None,
        help='Path to custom scenarios file (optional)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Run appropriate mode
    if args.mode == 'production':
        run_production(args)
    
    elif args.mode == 'fixed':
        run_fixed(args)
    
    elif args.mode == 'benchmark':
        run_benchmark(args)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
