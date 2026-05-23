"""Image processing utilities for detection."""

import numpy as np
from PIL import Image, ImageDraw

from .config import CONF_THRESHOLD, IOU_THRESHOLD, INPUT_HW

CLASS_NAMES = {
    0: "foodie_noodles_olympics",
    1: "mr_noodles_competitor",
}


def set_class_names(class_names: dict[int, str]) -> None:
    """Update class names from model metadata."""
    if class_names:
        CLASS_NAMES.clear()
        CLASS_NAMES.update(class_names)


class DetectionResult:
    """Represents a single detection result."""
    def __init__(self, class_id: int, confidence: float, box_xyxy: list[float]):
        self.class_id = class_id
        self.confidence = confidence
        self.box_xyxy = box_xyxy


def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    """Convert boxes from XYWH format to XYXY format."""
    out = np.zeros_like(boxes_xywh)
    out[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
    out[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
    out[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
    out[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
    return out


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute Intersection over Union (IoU) between a box and multiple boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area_box = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])

    union = area_box + area_boxes - intersection + 1e-9
    return intersection / union


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Non-Maximum Suppression (NMS) to filter overlapping detections."""
    order = np.argsort(scores)[::-1]
    keep: list[int] = []

    while order.size > 0:
        idx = int(order[0])
        keep.append(idx)
        if order.size == 1:
            break

        rest = order[1:]
        ious = compute_iou(boxes[idx], boxes[rest])
        order = rest[ious <= iou_threshold]

    return keep


def preprocess(img_rgb: np.ndarray, input_hw: tuple[int, int] = INPUT_HW) -> np.ndarray:
    """Preprocess image for ONNX model inference."""
    h, w = input_hw
    pil = Image.fromarray(img_rgb)
    resized = pil.resize((w, h))
    arr = np.array(resized).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr


def postprocess(
    output: np.ndarray,
    orig_w: int,
    orig_h: int,
    conf_threshold: float = CONF_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
    input_hw: tuple[int, int] = INPUT_HW,
) -> list[DetectionResult]:
    """Postprocess ONNX model output to get detections."""
    preds = np.squeeze(output)

    if preds.ndim == 2 and preds.shape[0] < preds.shape[1]:
        preds = preds.T

    if preds.ndim != 2 or preds.shape[1] < 6:
        return []

    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    mask = scores >= conf_threshold
    if not np.any(mask):
        return []

    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask]
    scores = scores[mask]

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)

    scale_x = orig_w / float(input_hw[1])
    scale_y = orig_h / float(input_hw[0])
    boxes_xyxy[:, [0, 2]] *= scale_x
    boxes_xyxy[:, [1, 3]] *= scale_y

    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

    keep = nms(boxes_xyxy, scores, iou_threshold)

    results: list[DetectionResult] = []
    for idx in keep:
        box = boxes_xyxy[idx].tolist()
        results.append(
            DetectionResult(
                class_id=int(class_ids[idx]),
                confidence=float(scores[idx]),
                box_xyxy=[float(v) for v in box],
            )
        )
    return results


def draw_boxes(img_rgb: np.ndarray, detections: list[DetectionResult]) -> np.ndarray:
    """Draw bounding boxes on the image."""
    img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img)
    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy
        label = get_class_name(det.class_id)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        draw.text((x1 + 2, max(0, y1 - 12)), f"{label} {det.confidence:.2f}", fill=(255, 0, 0))
    return np.array(img)


def get_class_name(class_id: int) -> str:
    """Get class name from class ID."""
    return CLASS_NAMES.get(class_id, f"class_{class_id}")
