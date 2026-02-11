"""VLM Backend Factory - Pluggable architecture for VLM backends"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class VLMBackendFactory:
    """
    Factory for creating VLM backend instances.
    Supports both embedded OpenVINO GenAI and OVMS backends.
    """
    
    @staticmethod
    def create_backend(
        backend_type: str,
        config: dict,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ):
        """
        Create VLM backend instance based on type.

        Args:
            backend_type: 'embedded' or 'ovms'
            config: Configuration dict with backend-specific settings
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            VLM backend instance (VLMPipeline or OVMSVLMClient)
        """
        backend_type = backend_type.lower()
        
        if backend_type == "ovms":
            return VLMBackendFactory._create_ovms_backend(config, max_new_tokens, temperature)
        elif backend_type == "embedded":
            return VLMBackendFactory._create_embedded_backend(config, max_new_tokens, temperature)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}. Use 'embedded' or 'ovms'")

    @staticmethod
    def _create_ovms_backend(config: dict, max_new_tokens: int, temperature: float):
        """Create OVMS backend."""
        from ovms_client import OVMSVLMClient, MockGenerationConfig
        
        endpoint = config.get("ovms_endpoint", "http://ovms-vlm:8000")
        model_name = config.get("ovms_model", "Qwen/Qwen2-VL-2B-Instruct")
        timeout = config.get("timeout_sec", 120)
        
        logger.info(f"[BACKEND-FACTORY] Creating OVMS backend: {endpoint}")
        
        client = OVMSVLMClient(
            endpoint=endpoint,
            model_name=model_name,
            timeout=timeout,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        # Return tuple: (client, generation_config)
        gen_config = MockGenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False
        )
        
        return client, gen_config

    @staticmethod
    def _create_embedded_backend(config: dict, max_new_tokens: int, temperature: float):
        """Create embedded OpenVINO GenAI backend."""
        try:
            import openvino as ov
            from openvino_genai import VLMPipeline, GenerationConfig
        except ImportError as e:
            logger.error("[BACKEND-FACTORY] openvino_genai not installed. Install with: pip install openvino-genai")
            raise ImportError("openvino_genai is required for embedded backend") from e
        
        model_path = config.get("model_path", "/model/Qwen2.5-VL-7B-Instruct-ov-int8")
        device = config.get("device", "GPU")
        
        logger.info(f"[BACKEND-FACTORY] Creating embedded VLM backend: {model_path} on {device}")
        
        # Initialize OpenVINO Core
        core = ov.Core()
        if device.upper().startswith("GPU"):
            core.set_property("GPU", {"GPU_THROUGHPUT_STREAMS": "1"})
        
        # Load VLM Pipeline
        pipeline = VLMPipeline(models_path=model_path, device=device)
        
        # Create generation config
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False
        )
        
        logger.info("[BACKEND-FACTORY] Embedded VLM backend loaded successfully")
        
        return pipeline, gen_config

    @staticmethod
    def get_backend_type_from_env() -> str:
        """
        Get backend type from environment variable.
        
        Returns:
            'embedded' or 'ovms'
        """
        return os.getenv("VLM_BACKEND", "embedded").lower()
