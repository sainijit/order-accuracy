"""
Frame Pipeline for Parallel Pipeline - GStreamer Python Callback

This module provides the gvapython callback function for processing video frames
in the persistent pipeline. Implements:
1. OCR-based order ID detection
2. Order change detection with finalization
3. Frame upload to MinIO
4. 10-second timeout for auto-finalization
5. EOS marker writing

Based on application-service/app/frame_pipeline.py but adapted for parallel-pipeline.
"""

import os
import sys
import time
import logging
import re
from typing import Optional
from io import BytesIO

# Import required libraries
try:
    import cv2
    import numpy as np
    from minio import Minio
    from minio.error import S3Error
    import easyocr
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}", file=sys.stderr)
    print("Make sure opencv-python, numpy, minio, and easyocr are installed", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Global State (persists across frame callbacks)
# ============================================================================
_reader = None  # EasyOCR reader instance
_minio_client = None  # MinIO client instance
_current_order_id: Optional[str] = None  # Current active order
_order_frame_count = 0  # Frame counter for current order
_frame_counter = 0  # Global frame counter
_last_order_time: Optional[float] = None  # Last time we saw an order
_station_id: Optional[str] = None  # Station identifier
_bucket_name: Optional[str] = None  # MinIO bucket name

# Configuration from environment
ORDER_TIMEOUT_SECONDS = 10  # Auto-finalize after 10 seconds without order


def _initialize_components():
    """Initialize OCR reader and MinIO client (called once)."""
    global _reader, _minio_client, _station_id, _bucket_name
    
    if _reader is None:
        logger.info("Initializing EasyOCR reader...")
        _reader = easyocr.Reader(['en'], gpu=False)
        logger.info("EasyOCR reader initialized")
    
    if _minio_client is None:
        _station_id = os.environ.get('STATION_ID', 'station_unknown')
        _bucket_name = os.environ.get('MINIO_BUCKET', 'frames')
        endpoint = os.environ.get('MINIO_ENDPOINT', 'minio:9000')
        
        logger.info(f"Initializing MinIO client (endpoint: {endpoint}, bucket: {_bucket_name})")
        _minio_client = Minio(
            endpoint,
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False
        )
        
        # Ensure bucket exists
        try:
            if not _minio_client.bucket_exists(_bucket_name):
                _minio_client.make_bucket(_bucket_name)
                logger.info(f"Created MinIO bucket: {_bucket_name}")
            else:
                logger.info(f"MinIO bucket exists: {_bucket_name}")
        except S3Error as e:
            logger.error(f"MinIO bucket check failed: {e}")


def safe_get_image(frame) -> Optional[np.ndarray]:
    """
    Safely extract image from GStreamer VideoFrame.
    
    Args:
        frame: GStreamer VideoFrame object
        
    Returns:
        numpy array (BGR) or None if extraction fails
    """
    try:
        # Method 1: Direct data access
        if hasattr(frame, 'data') and frame.data is not None:
            width = frame.video_info().width
            height = frame.video_info().height
            
            # VideoFrame.data() returns bytes
            frame_data = frame.data()
            if frame_data:
                # Convert bytes to numpy array
                img_array = np.frombuffer(frame_data, dtype=np.uint8)
                # Reshape to (height, width, 3) for BGR
                img = img_array.reshape((height, width, 3))
                return img
        
        # Method 2: Use video_frame() if available
        if hasattr(frame, 'video_frame'):
            vf = frame.video_frame()
            if vf:
                return vf
        
        logger.warning("Could not extract image from frame")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting image: {e}")
        return None


def detect_order_id(image: np.ndarray) -> Optional[str]:
    """
    Detect order ID using OCR.
    
    Looks for pattern: #<digits> or ORDER_<digits>
    
    Args:
        image: BGR image
        
    Returns:
        Order ID string or None
    """
    global _reader
    
    if _reader is None:
        return None
    
    try:
        # Run OCR
        results = _reader.readtext(image)
        
        # Search for order ID pattern
        for (bbox, text, confidence) in results:
            # Pattern 1: #123 or #9253
            match = re.search(r'#(\d+)', text)
            if match and confidence > 0.5:
                order_id = match.group(1)
                logger.info(f"Detected order ID: {order_id} (confidence: {confidence:.2f})")
                return order_id
            
            # Pattern 2: ORDER_123
            match = re.search(r'ORDER[_\s]*(\d+)', text, re.IGNORECASE)
            if match and confidence > 0.5:
                order_id = match.group(1)
                logger.info(f"Detected order ID: {order_id} (confidence: {confidence:.2f})")
                return order_id
        
        return None
        
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return None


def upload_frame(order_id: str, frame_idx: int, image: np.ndarray) -> bool:
    """
    Upload frame to MinIO.
    
    Structure: {STATION_ID}/{order_id}/frame_{idx}.jpg
    
    Args:
        order_id: Order identifier
        frame_idx: Frame index (0, 1, 2, ...)
        image: BGR image array
        
    Returns:
        True if successful, False otherwise
    """
    global _minio_client, _station_id, _bucket_name
    
    if _minio_client is None or _station_id is None:
        logger.error("MinIO client not initialized")
        return False
    
    try:
        # Encode image as JPEG
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_bytes = buffer.tobytes()
        
        # Create object path
        object_name = f"{_station_id}/{order_id}/frame_{frame_idx}.jpg"
        
        # Upload to MinIO
        _minio_client.put_object(
            _bucket_name,
            object_name,
            BytesIO(image_bytes),
            length=len(image_bytes),
            content_type='image/jpeg'
        )
        
        logger.debug(f"Uploaded: {object_name}")
        return True
        
    except Exception as e:
        logger.error(f"Frame upload failed: {e}")
        return False


def finalize_order(order_id: str):
    """
    Finalize order by writing EOS marker.
    
    Structure: {STATION_ID}/{order_id}/__EOS__
    
    Args:
        order_id: Order identifier
    """
    global _minio_client, _station_id, _bucket_name, _order_frame_count
    
    if _minio_client is None or _station_id is None:
        logger.error("MinIO client not initialized")
        return
    
    try:
        # Create EOS marker
        eos_marker = f"Order {order_id} completed with {_order_frame_count} frames"
        eos_bytes = eos_marker.encode('utf-8')
        
        # Write to MinIO
        object_name = f"{_station_id}/{order_id}/__EOS__"
        _minio_client.put_object(
            _bucket_name,
            object_name,
            BytesIO(eos_bytes),
            length=len(eos_bytes),
            content_type='text/plain'
        )
        
        logger.info(
            f"[{_station_id}] Finalized order {order_id} "
            f"with {_order_frame_count} frames (EOS marker written)"
        )
        
    except Exception as e:
        logger.error(f"Failed to write EOS marker: {e}")


def check_timeout():
    """
    Check if current order has timed out (10 seconds without new frames).
    
    If timeout detected, auto-finalize the current order.
    """
    global _current_order_id, _last_order_time
    
    if _current_order_id is not None and _last_order_time is not None:
        elapsed = time.time() - _last_order_time
        
        if elapsed > ORDER_TIMEOUT_SECONDS:
            logger.info(
                f"Order timeout detected: {_current_order_id} "
                f"({elapsed:.1f}s since last frame)"
            )
            finalize_order(_current_order_id)
            _current_order_id = None
            _last_order_time = None


def process_frame(frame):
    """
    GStreamer Python callback - called for each video frame.
    
    This is the main entry point called by gvapython element.
    
    Flow:
    1. Initialize components (first call only)
    2. Extract image from frame
    3. Run OCR to detect order ID
    4. Check for timeout (10 seconds)
    5. Detect order change
    6. Upload frame to MinIO
    7. Update state
    
    Args:
        frame: GStreamer VideoFrame object
        
    Returns:
        True to continue pipeline
    """
    global _current_order_id, _order_frame_count, _frame_counter, _last_order_time
    
    # Initialize on first call
    if _reader is None or _minio_client is None:
        _initialize_components()
    
    _frame_counter += 1
    
    # Extract image
    image = safe_get_image(frame)
    if image is None:
        logger.warning(f"Frame {_frame_counter}: Failed to extract image")
        return True
    
    # Run OCR
    detected_order_id = detect_order_id(image)
    
    # Check for timeout (10 seconds without new order)
    check_timeout()
    
    # ========================================================================
    # 4. Order Change Detection
    # ========================================================================
    if detected_order_id is not None:
        if detected_order_id != _current_order_id:
            # Order change detected
            if _current_order_id is not None:
                # Finalize previous order
                logger.info(
                    f"Order change: {_current_order_id} → {detected_order_id} "
                    f"(finalizing previous)"
                )
                finalize_order(_current_order_id)
            
            # Start new order
            logger.info(f"Starting new order: {detected_order_id}")
            _current_order_id = detected_order_id
            _order_frame_count = 0
            _last_order_time = time.time()
    
    # ========================================================================
    # 5. Upload frame
    # ========================================================================
    if _current_order_id is not None:
        upload_success = upload_frame(_current_order_id, _order_frame_count, image)
        
        if upload_success:
            _order_frame_count += 1
            _last_order_time = time.time()  # Update last seen time
            
            if _order_frame_count % 10 == 0:
                logger.info(
                    f"Order {_current_order_id}: "
                    f"{_order_frame_count} frames uploaded"
                )
    
    return True  # Continue pipeline


# ============================================================================
# Module initialization
# ============================================================================
logger.info("frame_pipeline module loaded (parallel-pipeline version)")
