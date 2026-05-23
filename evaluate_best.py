"""Evaluate the current best YOLO checkpoint on the stratified test split."""

from __future__ import annotations

from ultralytics import YOLO


def main() -> None:
    model = YOLO("runs/train/pipeline_run/weights/best.pt")
    results = model.val(
        data="dataset/dataset_stratified/data.yaml",
        split="test",
        device="cpu",
        workers=0,
        verbose=True,
        plots=True,
    )

    metrics = results.results_dict
    print(
        "EVAL_METRICS "
        f"precision={metrics['metrics/precision(B)']:.6f} "
        f"recall={metrics['metrics/recall(B)']:.6f} "
        f"mAP50={metrics['metrics/mAP50(B)']:.6f} "
        f"mAP50_95={metrics['metrics/mAP50-95(B)']:.6f}"
    )


if __name__ == "__main__":
    main()
