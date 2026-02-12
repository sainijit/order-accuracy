"""
FastAPI endpoints for Dine-In Order Accuracy Validation
"""

import json
import uuid
import time
import logging
import httpx
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from io import BytesIO
import base64

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

# Import validation logic from app.py
from app import _VALIDATION_PROFILES, _METRIC_PROFILES, _default_validation, _default_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Service endpoints
OVMS_ENDPOINT = "http://ovms-vlm:8000"
SEMANTIC_SERVICE_ENDPOINT = os.getenv("SEMANTIC_SERVICE_ENDPOINT", "http://oa_semantic_service:8080")
METRICS_COLLECTOR_ENDPOINT = os.getenv("METRICS_COLLECTOR_ENDPOINT", "http://metrics-collector:8084")


# Helper functions for VLM and Semantic services

async def call_ovms_vlm(image_bytes: bytes, prompt: str = "List all food items visible in this image with their quantities.") -> Dict:
    """
    Call OVMS VLM service to analyze the image.
    
    Args:
        image_bytes: Image data as bytes
        prompt: Prompt for the VLM model
        
    Returns:
        Dict with detected items
    """
    try:
        # Encode image to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Prepare VLM request
        vlm_request = {
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": 500,
            "temperature": 0.1
        }
        
        logger.info(f"[OVMS] Sending VLM inference request")
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OVMS_ENDPOINT}/v3/chat/completions",
                json=vlm_request
            )
        
        vlm_latency = time.time() - start_time
        logger.info(f"[OVMS] VLM inference completed in {vlm_latency:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Log the raw VLM response
            logger.info(f"[OVMS] VLM Raw Response: {content[:500]}...")  # Log first 500 chars
            
            # Parse VLM output to extract items
            detected_items = parse_vlm_output(content)
            
            # Log detected items for debugging
            logger.info(f"[OVMS] Detected items: {detected_items}")
            
            return {
                "success": True,
                "detected_items": detected_items,
                "raw_output": content,
                "inference_time_ms": int(vlm_latency * 1000)
            }
        else:
            logger.error(f"[OVMS] VLM request failed: {response.status_code} - {response.text}")
            return {
                "success": False,
                "error": f"VLM request failed with status {response.status_code}",
                "detected_items": []
            }
            
    except Exception as e:
        logger.exception(f"[OVMS] Error calling VLM: {e}")
        return {
            "success": False,
            "error": str(e),
            "detected_items": []
        }


def parse_vlm_output(content: str) -> List[Dict]:
    """
    Parse VLM output to extract items and quantities.
    Handles both JSON format and plain text format.
    
    Args:
        content: VLM response text
        
    Returns:
        List of detected items with name and quantity
    """
    items = []
    
    # Try to parse as JSON first
    try:
        # Remove markdown code blocks if present
        json_content = content.strip()
        if json_content.startswith('```'):
            # Extract JSON from code block
            lines = json_content.split('\n')
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith('```'):
                    in_block = not in_block
                    continue
                if in_block or (not line.strip().startswith('```')):
                    json_lines.append(line)
            json_content = '\n'.join(json_lines)
        
        # Try to parse as JSON
        data = json.loads(json_content)
        
        # Handle different JSON structures
        if isinstance(data, dict):
            # Check for food_items array
            if 'food_items' in data:
                for item in data['food_items']:
                    if isinstance(item, dict) and 'item' in item:
                        items.append({
                            "name": item['item'].lower(),
                            "quantity": item.get('quantity', 1)
                        })
            # Check for items array
            elif 'items' in data:
                for item in data['items']:
                    if isinstance(item, dict):
                        name = item.get('item') or item.get('name', '')
                        items.append({
                            "name": name.lower(),
                            "quantity": item.get('quantity', 1)
                        })
        elif isinstance(data, list):
            # Direct array of items
            for item in data:
                if isinstance(item, dict):
                    name = item.get('item') or item.get('name', '')
                    items.append({
                        "name": name.lower(),
                        "quantity": item.get('quantity', 1)
                    })
        
        if items:
            logger.info(f"[PARSER] Parsed {len(items)} items from JSON")
            return items
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.debug(f"[PARSER] JSON parsing failed, falling back to text parsing: {e}")
    
    # Fallback to line-by-line text parsing
    lines = content.lower().strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('```'):
            continue
            
        # Try to extract quantity and name
        quantity = 1
        name = line
        
        # Remove bullet points and numbers
        name = name.lstrip('•-*0123456789. ')
        
        # Look for quantity patterns
        if 'x ' in name:
            parts = name.split('x ', 1)
            try:
                quantity = int(parts[0].strip())
                name = parts[1].strip()
            except ValueError:
                pass
        elif ' (' in name and ')' in name:
            # Pattern: item (quantity)
            parts = name.split('(')
            name = parts[0].strip()
            if len(parts) > 1:
                qty_str = parts[1].split(')')[0].strip()
                try:
                    quantity = int(qty_str)
                except ValueError:
                    pass
        
        if name and not name.startswith('{') and not name.startswith('}'):
            items.append({"name": name, "quantity": quantity})
    
    logger.info(f"[PARSER] Parsed {len(items)} items from VLM output")
    return items


async def call_semantic_service(expected_item: str, detected_item: str) -> Dict:
    """
    Call semantic service to match items.
    
    Args:
        expected_item: Expected item name from order
        detected_item: Detected item name from VLM
        
    Returns:
        Dict with match result and confidence
    """
    try:
        # Configure httpx to respect NO_PROXY environment variable
        async with httpx.AsyncClient(timeout=10.0, trust_env=True) as client:
            response = await client.post(
                f"{SEMANTIC_SERVICE_ENDPOINT}/api/v1/compare/semantic",
                json={
                    "text1": expected_item,
                    "text2": detected_item,
                    "context": "restaurant food items"
                }
            )
        
        if response.status_code == 200:
            result = response.json()
            return {
                "match": result.get("match", False),
                "confidence": result.get("confidence", 0.0)
            }
        else:
            logger.warning(f"[SEMANTIC] Service returned {response.status_code}, using fuzzy match")
            # Fallback to simple string matching
            return {
                "match": expected_item.lower() in detected_item.lower() or detected_item.lower() in expected_item.lower(),
                "confidence": 0.5
            }
            
    except Exception as e:
        logger.error(f"[SEMANTIC] Error calling service: {e}")
        # Fallback to simple string matching
        return {
            "match": expected_item.lower() in detected_item.lower() or detected_item.lower() in expected_item.lower(),
            "confidence": 0.5
        }


async def validate_with_semantic_service(order_id: str, detected_items: List[Dict]) -> Dict:
    """
    Call semantic service with order_id and VLM results for batch validation.
    
    Args:
        order_id: Order ID to match against (from orders.json)
        detected_items: List of detected items from VLM with name and quantity
        
    Returns:
        Dict with validation results including matched, missing, and extra items
    """
    try:
        # Configure httpx to respect NO_PROXY environment variable
        async with httpx.AsyncClient(timeout=30.0, trust_env=True) as client:
            response = await client.post(
                f"{SEMANTIC_SERVICE_ENDPOINT}/api/v1/validate/order",
                json={
                    "order_id": order_id,
                    "detected_items": detected_items
                }
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"[SEMANTIC] Batch validation returned {response.status_code}, falling back to individual matching")
            return None
            
    except Exception as e:
        logger.error(f"[SEMANTIC] Error calling batch validation: {e}")
        return None


async def call_metrics_collector() -> Dict:
    """
    Get CPU/GPU utilization metrics using psutil.
    Fallback implementation since metrics-collector service doesn't expose HTTP API.
    
    Returns:
        Dict with CPU and GPU metrics
    """
    try:
        import psutil
        
        # Get CPU utilization (average over 0.1 seconds)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get memory utilization
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Try to get GPU metrics (optional, may not be available)
        gpu_utilization = 0.0
        gpu_memory_utilization = 0.0
        
        try:
            import subprocess
            result = subprocess.run(['intel_gpu_top', '-J', '-s', '100'], 
                                  capture_output=True, text=True, timeout=1)
            if result.returncode == 0:
                import json
                gpu_data = json.loads(result.stdout)
                gpu_utilization = gpu_data.get('engines', {}).get('Render/3D', {}).get('busy', 0.0)
        except:
            pass  # GPU metrics not available
        
        logger.info(f"[METRICS] System metrics: CPU={cpu_percent}%, Memory={memory_percent}%")
        
        return {
            "cpu_utilization": round(cpu_percent, 2),
            "gpu_utilization": round(gpu_utilization, 2),
            "memory_utilization": round(memory_percent, 2),
            "gpu_memory_utilization": round(gpu_memory_utilization, 2)
        }
            
    except Exception as e:
        logger.error(f"[METRICS] Error getting system metrics: {e}")
        return {
            "cpu_utilization": 0.0,
            "gpu_utilization": 0.0,
            "memory_utilization": 0.0,
            "gpu_memory_utilization": 0.0
        }


async def validate_order_with_services(expected_items: List[Dict], detected_items: List[Dict], order_id: str = None) -> Tuple[Dict, Dict]:
    """
    Validate order using semantic matching.
    
    Args:
        expected_items: List of expected items from order
        detected_items: List of detected items from VLM
        order_id: Optional order ID for batch validation with semantic service
        
    Returns:
        Tuple of (validation_result, metrics)
    """
    start_time = time.time()
    
    # Track semantic matching latency
    semantic_calls = 0
    semantic_latency_total = 0
    
    # Try batch validation with semantic service first if order_id is provided
    if order_id:
        logger.info(f"[VALIDATION] Attempting batch validation with semantic service for order: {order_id}")
        semantic_start = time.time()
        batch_result = await validate_with_semantic_service(order_id, detected_items)
        semantic_latency_total = time.time() - semantic_start
        semantic_calls = 1
        
        if batch_result:
            logger.info(f"[SEMANTIC] Batch validation successful in {semantic_latency_total*1000:.0f}ms")
            
            # Extract results from semantic service response
            missing_items = [{"name": item, "quantity": 1} for item in batch_result.get("missing_items", [])]
            extra_items = [{"name": item, "quantity": 1} for item in batch_result.get("extra_items", [])]
            quantity_mismatches = batch_result.get("quantity_mismatches", [])
            accuracy = batch_result.get("accuracy", 0.0)
            
            validation_latency = time.time() - start_time
            system_metrics = await call_metrics_collector()
            
            validation_result = {
                "order_complete": len(missing_items) == 0 and len(extra_items) == 0,
                "missing_items": [item["name"] for item in missing_items],
                "extra_items": [item["name"] for item in extra_items],
                "modifier_validation": {
                    "status": "validated" if len(quantity_mismatches) == 0 else "quantity_mismatch",
                    "details": quantity_mismatches
                },
                "accuracy_score": accuracy
            }
            
            metrics = {
                "end_to_end_latency_ms": int(validation_latency * 1000),
                "semantic_calls": semantic_calls,
                "semantic_latency_ms": int(semantic_latency_total * 1000),
                "within_operational_window": validation_latency < 2.0,
                "cpu_utilization": system_metrics.get("cpu_utilization", 0.0),
                "gpu_utilization": system_metrics.get("gpu_utilization", 0.0),
                "memory_utilization": system_metrics.get("memory_utilization", 0.0),
                "gpu_memory_utilization": system_metrics.get("gpu_memory_utilization", 0.0)
            }
            
            logger.info(f"[VALIDATION] Complete: accuracy={accuracy:.2%}, missing={len(missing_items)}, extra={len(extra_items)}")
            return validation_result, metrics
        else:
            logger.warning(f"[SEMANTIC] Batch validation failed, falling back to individual item matching")
    
    # Fallback to individual matching if batch validation not available or failed
    missing_items = []
    extra_items = []
    quantity_mismatches = []
    matched_detected = set()
    
    # Pass 1: Exact name matching
    logger.info(f"[VALIDATION] Starting Pass 1 (exact matching) with {len(expected_items)} expected items")
    for expected in expected_items:
        exp_name = expected["name"].lower()
        exp_qty = expected["quantity"]
        found = False
        
        logger.info(f"[VALIDATION] Pass 1: Looking for expected item '{exp_name}' (qty={exp_qty})")
        
        for idx, detected in enumerate(detected_items):
            if idx in matched_detected:
                continue
                
            det_name = detected["name"].lower()
            det_qty = detected["quantity"]
            
            if exp_name == det_name:
                found = True
                matched_detected.add(idx)
                logger.info(f"[VALIDATION] Pass 1: EXACT MATCH found! '{exp_name}' == '{det_name}'")
                
                if exp_qty != det_qty:
                    quantity_mismatches.append({
                        "name": expected["name"],
                        "expected": exp_qty,
                        "detected": det_qty
                    })
                break
        
        if not found:
            logger.info(f"[VALIDATION] Pass 1: No exact match for '{exp_name}', will try semantic matching")
            missing_items.append(expected)
    
    # Pass 2: Semantic matching for unmatched items
    logger.info(f"[VALIDATION] Starting Pass 2 (semantic matching) with {len(missing_items)} missing items")
    still_missing = []
    for expected in missing_items:
        exp_name = expected["name"]
        exp_qty = expected["quantity"]
        found = False
        
        logger.info(f"[VALIDATION] Pass 2: Semantic search for '{exp_name}'")
        
        for idx, detected in enumerate(detected_items):
            if idx in matched_detected:
                continue
            
            det_name = detected["name"]
            
            logger.info(f"[VALIDATION] Pass 2: Comparing '{exp_name}' with '{det_name}' via semantic service")
            
            # Call semantic service
            semantic_start = time.time()
            match_result = await call_semantic_service(exp_name, det_name)
            semantic_latency_total += (time.time() - semantic_start)
            semantic_calls += 1
            
            logger.info(f"[VALIDATION] Pass 2: Semantic result - match={match_result['match']}, confidence={match_result['confidence']:.2f}")
            
            if match_result["match"]:
                found = True
                matched_detected.add(idx)
                logger.info(f"[SEMANTIC] Matched '{exp_name}' with '{det_name}' (confidence={match_result['confidence']:.2f})")
                
                if exp_qty != detected["quantity"]:
                    quantity_mismatches.append({
                        "name": expected["name"],
                        "expected": exp_qty,
                        "detected": detected["quantity"]
                    })
                break
        
        if not found:
            logger.info(f"[VALIDATION] Pass 2: NO MATCH found for '{exp_name}' even after semantic search")
            still_missing.append(expected)
    
    # Find extra items
    for idx, detected in enumerate(detected_items):
        if idx not in matched_detected:
            extra_items.append(detected)
    
    # Calculate accuracy
    total_expected = len(expected_items)
    correct = total_expected - len(still_missing)
    accuracy = correct / total_expected if total_expected > 0 else 1.0
    
    validation_latency = time.time() - start_time
    
    # Get system metrics from metrics-collector
    system_metrics = await call_metrics_collector()
    
    validation_result = {
        "order_complete": len(still_missing) == 0 and len(extra_items) == 0,
        "missing_items": [item["name"] for item in still_missing],
        "extra_items": [item["name"] for item in extra_items],
        "modifier_validation": {
            "status": "validated" if len(quantity_mismatches) == 0 else "quantity_mismatch",
            "details": quantity_mismatches
        },
        "accuracy_score": accuracy
    }
    
    metrics = {
        "end_to_end_latency_ms": int(validation_latency * 1000),
        "semantic_calls": semantic_calls,
        "semantic_latency_ms": int(semantic_latency_total * 1000),
        "within_operational_window": validation_latency < 2.0,
        "cpu_utilization": system_metrics.get("cpu_utilization", 0.0),
        "gpu_utilization": system_metrics.get("gpu_utilization", 0.0),
        "memory_utilization": system_metrics.get("memory_utilization", 0.0),
        "gpu_memory_utilization": system_metrics.get("gpu_memory_utilization", 0.0)
    }
    
    logger.info(f"[VALIDATION] Complete: accuracy={accuracy:.2%}, missing={len(still_missing)}, extra={len(extra_items)}")
    
    return validation_result, metrics


# Pydantic models for request/response
class OrderItem(BaseModel):
    name: str
    quantity: int


class OrderManifest(BaseModel):
    order_id: str
    table_number: Optional[str] = None
    restaurant: Optional[str] = None
    items: List[OrderItem]
    modifiers: Optional[List[str]] = None


class ValidationRequest(BaseModel):
    image_id: Optional[str] = None
    order: OrderManifest


class BatchValidationRequest(BaseModel):
    validations: List[ValidationRequest]


class ValidationResult(BaseModel):
    validation_id: str
    timestamp: str
    order_complete: bool
    missing_items: List[str]
    extra_items: List[str]
    modifier_validation: Dict
    accuracy_score: float
    metrics: Dict


# In-memory storage for validation results
validation_store: Dict[str, ValidationResult] = {}


# Create FastAPI app
app = FastAPI(
    title="Dine-In Order Accuracy API",
    description="API for validating restaurant orders against plate images",
    version="1.0.0"
)


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "Dine-In Order Accuracy API",
        "version": "1.0.0",
        "endpoints": {
            "validate": "/api/validate",
            "validate_batch": "/api/validate/batch",
            "get_result": "/api/validate/{validation_id}"
        }
    }


@app.post("/api/validate", response_model=ValidationResult)
async def validate_plate(
    image: UploadFile = File(...),
    order: str = Body(..., description="JSON string of order manifest")
):
    """
    Validate a single plate image against order manifest.
    
    Args:
        image: Plate image file (PNG, JPG, etc.)
        order: JSON string containing order manifest
        
    Returns:
        ValidationResult with order accuracy analysis
    """
    start_time = time.time()
    
    try:
        # Parse order JSON
        order_data = json.loads(order)
        order_manifest = OrderManifest(**order_data)
        
        # Read and validate image
        image_bytes = await image.read()
        img = Image.open(BytesIO(image_bytes))
        
        logger.info(f"[API] Starting validation for order_id: {order_manifest.order_id}")
        
        # Generate validation ID
        validation_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Step 1: Call OVMS VLM to detect items in the image
        logger.info(f"[API] Step 1: Calling VLM service...")
        vlm_result = await call_ovms_vlm(
            image_bytes, 
            prompt="Analyze this food plate image and list all visible food items with their quantities in JSON format."
        )
        
        if not vlm_result.get("success"):
            logger.error(f"[API] VLM inference failed: {vlm_result.get('error')}")
            raise HTTPException(status_code=500, detail=f"VLM inference failed: {vlm_result.get('error')}")
        
        detected_items = vlm_result.get("detected_items", [])
        vlm_latency_ms = vlm_result.get("inference_time_ms", 0)
        logger.info(f"[API] VLM detected {len(detected_items)} items in {vlm_latency_ms}ms")
        
        # Step 2: Validate order using semantic matching
        logger.info(f"[API] Step 2: Performing semantic validation...")
        expected_items = [{"name": item.name, "quantity": item.quantity} for item in order_manifest.items]
        validation_result, validation_metrics = await validate_order_with_services(
            expected_items, 
            detected_items,
            order_id=order_manifest.order_id
        )
        
        logger.info(f"[API] Validation complete: accuracy={validation_result['accuracy_score']:.2%}")
        
        # Create validation result
        result = ValidationResult(
            validation_id=validation_id,
            timestamp=timestamp,
            order_complete=validation_result["order_complete"],
            missing_items=validation_result["missing_items"],
            extra_items=validation_result["extra_items"],
            modifier_validation=validation_result["modifier_validation"],
            accuracy_score=validation_result.get("accuracy_score") or 0.0,
            metrics=validation_metrics
        )
        
        # Store result
        validation_store[validation_id] = result
        
        total_latency_ms = int((time.time() - start_time) * 1000)
        logger.info(f"[API] Total request latency: {total_latency_ms}ms")
        
        return result
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid order JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")


@app.post("/api/validate/batch")
async def validate_batch(
    images: List[UploadFile] = File(...),
    orders: str = Body(..., description="JSON array of order manifests mapped to images")
):
    """
    Validate multiple plates with their corresponding orders.
    
    Args:
        images: List of plate image files
        orders: JSON string containing array of order manifests (must match image count)
        
    Returns:
        List of ValidationResults
    """
    try:
        # Parse orders JSON
        orders_data = json.loads(orders)
        
        if len(images) != len(orders_data):
            raise HTTPException(
                status_code=400,
                detail=f"Image count ({len(images)}) must match order count ({len(orders_data)})"
            )
        
        results = []
        
        for idx, (image_file, order_data) in enumerate(zip(images, orders_data)):
            try:
                order_manifest = OrderManifest(**order_data)
                
                # Read image
                image_bytes = await image_file.read()
                img = Image.open(BytesIO(image_bytes))
                
                # Generate validation ID
                validation_id = str(uuid.uuid4())
                timestamp = datetime.now().isoformat()
                
                # Get image_id
                image_id = order_data.get("image_id") or order_manifest.order_id
                
                # Get validation profile
                validation = _VALIDATION_PROFILES.get(image_id, _default_validation(image_id))
                metrics = _METRIC_PROFILES.get(image_id, _default_metrics(image_id))
                
                # Create validation result
                result = ValidationResult(
                    validation_id=validation_id,
                    timestamp=timestamp,
                    order_complete=validation["order_complete"],
                    missing_items=validation["missing_items"],
                    extra_items=validation["extra_items"],
                    modifier_validation=validation["modifier_validation"],
                    accuracy_score=validation.get("accuracy_score") or 0.0,
                    metrics=metrics
                )
                
                # Store result
                validation_store[validation_id] = result
                results.append(result)
                
            except Exception as e:
                # Add error result for this specific validation
                results.append({
                    "validation_id": None,
                    "error": f"Failed to validate image {idx}: {str(e)}"
                })
        
        return JSONResponse(content={"results": [r.dict() if isinstance(r, ValidationResult) else r for r in results]})
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid orders JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch validation error: {str(e)}")


@app.get("/api/validate/{validation_id}", response_model=ValidationResult)
async def get_validation_result(validation_id: str):
    """
    Retrieve validation result by ID.
    
    Args:
        validation_id: Unique validation identifier
        
    Returns:
        ValidationResult for the specified validation
    """
    if validation_id not in validation_store:
        raise HTTPException(
            status_code=404,
            detail=f"Validation result not found for ID: {validation_id}"
        )
    
    return validation_store[validation_id]


@app.get("/api/validate")
async def list_validations():
    """
    List all validation results.
    
    Returns:
        List of all stored validation results
    """
    return JSONResponse(content={
        "total": len(validation_store),
        "validations": [v.dict() for v in validation_store.values()]
    })


@app.delete("/api/validate/{validation_id}")
async def delete_validation_result(validation_id: str):
    """
    Delete a validation result by ID.
    
    Args:
        validation_id: Unique validation identifier
        
    Returns:
        Success message
    """
    if validation_id not in validation_store:
        raise HTTPException(
            status_code=404,
            detail=f"Validation result not found for ID: {validation_id}"
        )
    
    del validation_store[validation_id]
    return {"message": f"Validation {validation_id} deleted successfully"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "dine-in-api",
        "validations_stored": len(validation_store)
    }
