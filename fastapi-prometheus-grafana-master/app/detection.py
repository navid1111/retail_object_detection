"""Detection service for ONNX inference."""

import ast
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .config import CONF_THRESHOLD, INPUT_HW, IOU_THRESHOLD, MODEL_PATH
from .utils import DetectionResult, postprocess, preprocess, set_class_names


class DetectionService:
    """Service for running ONNX model inference."""
    
    def __init__(self):
        self.onnx_session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.input_hw: tuple[int, int] = INPUT_HW
        self.model_name: str = ""
        self.class_names: dict[int, str] = {}

    def initialize(self, model_path: str = MODEL_PATH) -> None:
        """Initialize the ONNX model."""
        if not Path(model_path).exists():
            raise RuntimeError(f"ONNX model not found at: {model_path}")

        self.onnx_session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        input_info = self.onnx_session.get_inputs()[0]
        self.input_name = input_info.name
        self.model_name = Path(model_path).name
        self.class_names = self._load_class_names()
        set_class_names(self.class_names)

        shape = input_info.shape
        if isinstance(shape, list) and len(shape) >= 4:
            h = shape[2] if isinstance(shape[2], int) else 640
            w = shape[3] if isinstance(shape[3], int) else 640
            self.input_hw = (int(h), int(w))

    def _load_class_names(self) -> dict[int, str]:
        """Read class-name metadata from the exported Ultralytics ONNX model."""
        if self.onnx_session is None:
            return {}

        raw_names = self.onnx_session.get_modelmeta().custom_metadata_map.get("names")
        if not raw_names:
            return {}

        try:
            parsed = ast.literal_eval(raw_names)
        except (SyntaxError, ValueError):
            return {}

        if not isinstance(parsed, dict):
            return {}

        return {int(class_id): str(name) for class_id, name in parsed.items()}

    def is_ready(self) -> bool:
        """Check if the model is initialized."""
        return self.onnx_session is not None and self.input_name is not None

    def infer(
        self,
        image_np: np.ndarray,
        conf_threshold: float = CONF_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
    ) -> tuple[list[DetectionResult], float]:
        """
        Run inference on the image.
        
        Returns:
            Tuple of (detections, inference_time_ms)
        """
        if not self.is_ready():
            raise RuntimeError("Model is not initialized")

        # Preprocess
        model_input = preprocess(image_np, self.input_hw)

        # Inference
        start = time.perf_counter()
        outputs = self.onnx_session.run(None, {self.input_name: model_input})
        inference_ms = (time.perf_counter() - start) * 1000.0

        # Postprocess
        orig_h, orig_w = image_np.shape[:2]
        detections = postprocess(
            outputs[0],
            orig_w=orig_w,
            orig_h=orig_h,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            input_hw=self.input_hw,
        )

        return detections, inference_ms
