"""
Validation agent for order comparison.
Supports both local semantic matching and external semantic-comparison-service.
"""

import os
import logging
from semantic_matcher import semantic_match
from semantic_client import SemanticComparisonClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize semantic service client if enabled
SEMANTIC_SERVICE_ENDPOINT = os.getenv("SEMANTIC_SERVICE_ENDPOINT", "http://semantic-service:8080")
USE_SEMANTIC_SERVICE = os.getenv("USE_SEMANTIC_SERVICE", "false").lower() == "true"

logger.info(f"Semantic service configuration: enabled={USE_SEMANTIC_SERVICE}, endpoint={SEMANTIC_SERVICE_ENDPOINT}")

semantic_client = None
if USE_SEMANTIC_SERVICE:
    semantic_client = SemanticComparisonClient(SEMANTIC_SERVICE_ENDPOINT)
    if semantic_client.health_check():
        logger.info(f"Semantic service connected successfully: {SEMANTIC_SERVICE_ENDPOINT}")
    else:
        logger.warning(f"Semantic service unavailable, falling back to local matching")
        semantic_client = None

def semantic_match_wrapper(vlm_pipeline, expected_name, detected_name):
    """
    Wrapper for semantic matching that tries external service first, then falls back to local.
    
    Args:
        vlm_pipeline: VLM pipeline for local semantic matching
        expected_name: Expected product name
        detected_name: Detected product name
    
    Returns:
        bool: True if items match semantically
    """
    logger.debug(f"[SEMANTIC] Matching: '{expected_name}' vs '{detected_name}'")
    
    # Try semantic service first if enabled and available
    if semantic_client:
        try:
            result = semantic_client.semantic_match(
                text1=expected_name,
                text2=detected_name,
                context="grocery products"
            )
            match_result = result["match"]
            confidence = result.get('confidence', 0.0)
            logger.info(f"[SEMANTIC-SERVICE] Match result: '{expected_name}' ↔ '{detected_name}' = {match_result} (confidence={confidence:.2f})")
            return match_result
        except Exception as e:
            logger.error(f"[SEMANTIC-SERVICE] Error occurred, falling back to local matching: {e}")
    
    # Fallback to local semantic matching
    logger.debug(f"[SEMANTIC-LOCAL] Using local matching for '{expected_name}' vs '{detected_name}'")
    result = semantic_match(vlm_pipeline, expected_name, detected_name)
    logger.info(f"[SEMANTIC-LOCAL] Match result: '{expected_name}' ↔ '{detected_name}' = {result}")
    return result

def validate_order(expected_items, detected_items, vlm_pipeline):
    logger.info(f"[VALIDATION] Starting order validation: {len(expected_items)} expected, {len(detected_items)} detected")
    logger.debug(f"[VALIDATION] Expected items: {expected_items}")
    logger.debug(f"[VALIDATION] Detected items: {detected_items}")
    
    missing = []
    extra = []
    quantity_mismatch = []
    matched_detected = set()

    # ---- Pass 1: exact match ----
    logger.debug("[VALIDATION] Pass 1: Exact name matching")
    for exp in expected_items:
        exp_name = exp["name"].lower()
        exp_qty = exp["quantity"]

        found = False
        for det in detected_items:
            det_name = det["name"].lower()
            det_qty = det["quantity"]

            if det_name == exp_name:
                found = True
                matched_detected.add(det_name)
                logger.debug(f"[VALIDATION] Exact match found: '{exp_name}'")
                if det_qty != exp_qty:
                    logger.info(f"[VALIDATION] Quantity mismatch: '{exp_name}' expected={exp_qty}, detected={det_qty}")
                    quantity_mismatch.append({
                        "name": exp_name,
                        "expected": exp_qty,
                        "detected": det_qty
                    })
                break

        if not found:
            logger.debug(f"[VALIDATION] No exact match for '{exp_name}', will try semantic matching")
            missing.append(exp)

    logger.info(f"[VALIDATION] Pass 1 complete: {len(missing)} items need semantic matching")

    # ---- Pass 2: semantic match on leftovers ----
    if missing:
        logger.debug("[VALIDATION] Pass 2: Semantic matching for unmatched items")
        still_missing = []

        for exp in missing:
            exp_name = exp["name"].lower()
            exp_qty = exp["quantity"]
            matched = False

            for det in detected_items:
                det_name = det["name"].lower()
                if det_name in matched_detected:
                    continue

                if semantic_match_wrapper(vlm_pipeline, exp_name, det_name):
                    matched = True
                    matched_detected.add(det_name)
                    logger.info(f"[VALIDATION] Semantic match found: '{exp_name}' matched with '{det_name}'")
                    if det["quantity"] != exp_qty:
                        logger.info(f"[VALIDATION] Quantity mismatch after semantic match: '{exp_name}' expected={exp_qty}, detected={det['quantity']}")
                        quantity_mismatch.append({
                            "name": exp_name,
                            "expected": exp_qty,
                            "detected": det["quantity"]
                        })
                    break

            if not matched:
                logger.warning(f"[VALIDATION] No match found for '{exp_name}' (exact or semantic)")
                still_missing.append(exp)

        missing = still_missing
        logger.info(f"[VALIDATION] Pass 2 complete: {len(missing)} items truly missing")

    # ---- Extras ----
    logger.debug("[VALIDATION] Checking for extra detected items")
    for det in detected_items:
        if det["name"].lower() not in matched_detected:
            logger.info(f"[VALIDATION] Extra item detected: '{det['name']}' (not in expected order)")
            extra.append(det)

    validation_result = {
        "missing": missing,
        "extra": extra,
        "quantity_mismatch": quantity_mismatch
    }
    
    logger.info(f"[VALIDATION] Complete: missing={len(missing)}, extra={len(extra)}, qty_mismatch={len(quantity_mismatch)}")
    logger.debug(f"[VALIDATION] Full result: {validation_result}")
    
    return validation_result
