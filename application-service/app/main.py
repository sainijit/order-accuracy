import cv2
import os
import logging
import time
from io import BytesIO
from ultralytics import YOLO
from minio import Minio
from fastapi import FastAPI, Body, UploadFile, File
from ocr_component import read_order_id
from vlm_service import run_vlm
from pipeline_runner import run_pipeline_async
from order_results import get_results
import uuid
import shutil

from config_loader import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

cfg = load_config()

MINIO = cfg["minio"]
BUCKETS = cfg["buckets"]

FRAMES_BUCKET = BUCKETS["frames"]
SELECTED_BUCKET = BUCKETS["selected"]


VIDEO_SOURCE = cfg["video"]["default_source"]
FPS_TARGET = cfg["video"]["fps_target"]


# =========================
# CONFIG
# =========================
# VIDEO_SOURCE = "/videos/sample.mp4"
CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.7
# FPS_TARGET = 1
HAND_LABELS = {"hand", "person"}
# BUCKET = "frames"

# =========================
# FASTAPI APP (API ONLY)
# =========================
app = FastAPI()

@app.post("/upload-video")
async def upload_and_run_video(file: UploadFile = File(...)):
    logger.info(f"Received video upload request: filename={file.filename}")
    
    if not file.filename.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
        logger.warning(f"Rejected unsupported file type: {file.filename}")
        return {
            "status": "error",
            "reason": "unsupported_file_type"
        }

    video_id = str(uuid.uuid4())
    save_path = f"/uploads/{video_id}_{file.filename}"
    logger.debug(f"Generated video_id={video_id}, save_path={save_path}")

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    logger.info(f"Video saved successfully: video_id={video_id}, path={save_path}")

    # Trigger pipeline
    logger.info(f"Triggering GStreamer pipeline for video_id={video_id}")
    run_pipeline_async(
        source_type="file",
        source=save_path
    )

    logger.info(f"Pipeline started for video_id={video_id}")
    return {
        "status": "started",
        "video_id": video_id,
        "path": save_path
    }


@app.post("/run-video")
def run_video(payload: dict = Body(...)):
    source_type = payload.get("source_type")  # file | rtsp | webcam | http
    source = payload.get("source")
    
    logger.info(f"Received run-video request: source_type={source_type}, source={source}")

    if not source_type or not source:
        logger.warning("Missing source_type or source in payload")
        return {
            "status": "error",
            "reason": "source_type_or_source_missing"
        }

    logger.info(f"Starting pipeline: source_type={source_type}, source={source}")
    run_pipeline_async(source_type, source)

    logger.info(f"Pipeline started successfully")
    return {
        "status": "started",
        "source_type": source_type,
        "source": source
    }


@app.get("/vlm/results")
def get_latest_vlm_results():
    logger.debug("Fetching latest VLM results")
    results = get_results()
    logger.info(f"Returning {len(results)} VLM results")
    return {
        "results": results
    }

@app.post("/run_vlm")
async def run_vlm_endpoint(payload: dict = Body(...)):
    order_id = payload.get("order_id")
    logger.info(f"[API] Received VLM processing request for order_id={order_id}")
    
    if not order_id:
        logger.warning("[API] Missing order_id in VLM request")
        return {
            "status": "error",
            "reason": "order_id_missing"
        }
    
    logger.debug(f"[API] Delegating order_id={order_id} to VLM service")
    result = await run_vlm(order_id)
    logger.info(f"[API] VLM processing completed for order_id={order_id}, status={result.get('status')}")
    return result

@app.post("/process_image")
async def process_image_endpoint(file: UploadFile = File(...)):
    """
    Direct image processing endpoint for testing.
    Accepts an image file and returns VLM detection results.
    """
    logger.info(f"[API] Received direct image processing request: filename={file.filename}")
    
    try:
        from vlm_service import vlm_instance
        import numpy as np
        from PIL import Image as PILImage
        
        # Read uploaded image
        contents = await file.read()
        img = PILImage.open(BytesIO(contents)).convert('RGB').resize((512, 512))
        img_array = np.array(img)
        
        # Process with VLM
        logger.info(f"[API] Processing image with VLM backend")
        start_time = time.time()
        result = vlm_instance.process([img_array])
        elapsed = time.time() - start_time
        
        logger.info(f"[API] Image processing completed: {len(result['items'])} items detected in {elapsed:.3f}s")
        
        return {
            "status": "success",
            "detected_items": result["items"],
            "num_frames": result["num_frames"],
            "inference_time_sec": result["inference_time_sec"],
            "total_time_sec": round(elapsed, 3)
        }
        
    except Exception as e:
        logger.error(f"[API] Image processing failed: {e}", exc_info=True)
        return {
            "status": "error",
            "reason": str(e)
        }

# =========================
# VIDEO PIPELINE (ISOLATED)
# =========================
def run_video_pipeline():
    logger.info("Initializing MinIO client")
    client = Minio(
        "minio:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False,
    )

    if not client.bucket_exists(BUCKET):
        logger.info(f"Creating bucket: {BUCKET}")
        client.make_bucket(BUCKET)

    logger.info("Loading YOLO model: yolov8n.pt")
    model = YOLO("yolov8n.pt")
    logger.info("YOLO model loaded successfully")

    logger.info(f"Opening video source: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {VIDEO_SOURCE}")
        raise RuntimeError("Video cannot be opened.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    skip_interval = max(1, int(fps / FPS_TARGET))
    frame_idx = 0

    logger.info(f"Processing video at {FPS_TARGET} FPS (skip_interval={skip_interval}, source_fps={fps})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip_interval != 0:
            frame_idx += 1
            continue

        detections = model.predict(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )[0]

        hand_present = False
        for box in detections.boxes:
            label = model.names[int(box.cls)].lower()
            if label in HAND_LABELS:
                hand_present = True
                break

        if hand_present:
            frame_idx += 1
            continue

        order_id = read_order_id(frame, frame_idx)
        if not order_id:
            frame_idx += 1
            continue

        _, jpeg = cv2.imencode(".jpg", frame)
        jpeg_bytes = jpeg.tobytes()

        objname = f"{order_id}/{frame_idx}.jpg"
        client.put_object(
            BUCKET,
            objname,
            data=BytesIO(jpeg_bytes),
            length=len(jpeg_bytes),
            content_type="image/jpeg",
        )

        logger.debug(f"Stored frame {frame_idx} for order_id={order_id} in bucket {BUCKET}")
        frame_idx += 1

    cap.release()
    logger.info(f"Video processing completed. Total frames processed: {frame_idx}")

# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    run_video_pipeline()
