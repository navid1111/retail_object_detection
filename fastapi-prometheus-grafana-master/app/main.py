"""FastAPI application for retail object detection."""

import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator

from .config import CONF_THRESHOLD, IOU_THRESHOLD, MODEL_PATH, UPLOAD_DIR
from .database import init_db
from .db_service import get_prediction, get_predictions, save_prediction
from .detection import DetectionService
from .metrics import (
    avg_confidence_gauge,
    confidence_histogram,
    detection_counter,
    detections_per_image,
    inference_time_histogram,
    low_confidence_counter,
    prediction_counter,
)
from .schemas import PredictionOut
from .utils import draw_boxes, get_class_name

# Initialize FastAPI app
app = FastAPI(title="Retail Detection API")

# Initialize detection service (global)
detection_service = DetectionService()


@app.on_event("startup")
def startup_event() -> None:
    """Initialize database and model on startup."""
    # Create upload directory at runtime (not import time) to avoid CI issues
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    detection_service.initialize(MODEL_PATH)


@app.get("/")
def home() -> dict:
    """Health check endpoint."""
    return {"message": "Retail Detection API is running"}


@app.get("/health")
def health() -> dict:
    """Detailed health check."""
    return {
        "status": "ok",
        "model": detection_service.model_name,
        "input_hw": detection_service.input_hw,
        "class_names": detection_service.class_names,
    }


@app.post("/predict", response_model=PredictionOut)
async def predict(
    image: UploadFile = File(...),
    ground_truth: str | None = Form(None),
) -> PredictionOut:
    """
    Run inference on an uploaded image.
    
    Args:
        image: Image file to analyze
        ground_truth: Optional ground truth JSON string
        
    Returns:
        PredictionOut with detections and metadata
    """
    if not detection_service.is_ready():
        raise HTTPException(status_code=500, detail="Model is not initialized")

    # Read and validate image
    try:
        image_bytes = await image.read()
        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_np = np.array(pil_img)
    orig_h, orig_w = image_np.shape[:2]

    # Save original image
    original_name = f"{uuid.uuid4().hex}_{image.filename or 'image.jpg'}"
    original_path = UPLOAD_DIR / original_name
    pil_img.save(original_path)

    # Run inference
    detections, inference_ms = detection_service.infer(
        image_np,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
    )

    # Draw boxes and save annotated image
    annotated = draw_boxes(image_np, detections)
    annotated_name = f"annotated_{original_name}"
    annotated_path = UPLOAD_DIR / annotated_name
    Image.fromarray(annotated).save(annotated_path)

    # Record Prometheus metrics
    model_name = detection_service.model_name
    prediction_counter.labels(model_name=model_name).inc()
    inference_time_histogram.labels(model_name=model_name).observe(inference_ms)
    detections_per_image.labels(model_name=model_name).observe(len(detections))

    # Calculate and record confidence metrics
    if detections:
        confidences = [det.confidence for det in detections]
        avg_conf = sum(confidences) / len(confidences)
        avg_confidence_gauge.labels(model_name=model_name).set(avg_conf)

        for det in detections:
            class_name = get_class_name(det.class_id)
            detection_counter.labels(class_name=class_name, model_name=model_name).inc()
            confidence_histogram.labels(class_name=class_name).observe(det.confidence)

            if det.confidence < 0.5:
                low_confidence_counter.labels(class_name=class_name, model_name=model_name).inc()

    # Save to database
    prediction = save_prediction(
        image_path=str(original_path),
        annotated_image_path=str(annotated_path),
        model_name=model_name,
        inference_ms=inference_ms,
        detections=detections,
        ground_truth_json=ground_truth,
    )

    return get_prediction(prediction.id)


@app.get("/predictions", response_model=list[PredictionOut])
def list_predictions(limit: int = 20) -> list[PredictionOut]:
    """Get recent predictions."""
    return get_predictions(limit)


@app.get("/predictions/{prediction_id}", response_model=PredictionOut)
def get_prediction_detail(prediction_id: int) -> PredictionOut:
    """Get a specific prediction by ID."""
    result = get_prediction(prediction_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return result


# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for uploaded images
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Add Prometheus instrumentation
Instrumentator().instrument(app).expose(app)
