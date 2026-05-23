import os
import sys
import yaml
from pathlib import Path

import wandb
from ultralytics import YOLO

# Support both `python -m src.model.train` and direct script execution.
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config import get_settings


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _metric_value(results, key: str, default: float = 0.0) -> float:
    return float(getattr(results, "results_dict", {}).get(key, default))


def _fitness_value(results) -> float:
    if hasattr(results, "best_fitness"):
        return float(results.best_fitness)

    fitness = getattr(results, "fitness", None)
    if callable(fitness):
        return float(fitness())
    if fitness is not None:
        return float(fitness)

    return _metric_value(results, "fitness", 0.0)


def _get_dvc_metadata(repo_root: str, dvc_file_path: str = "dataset.dvc") -> dict:
    """
    Load DVC file metadata (MD5 hash, file count) for dataset version tracking.
    
    Args:
        repo_root: Repository root path
        dvc_file_path: Path to .dvc file (relative to repo_root)
    
    Returns:
        Dict with 'md5_hash', 'nfiles', 'size' from the .dvc file, or empty dict if not found.
    """
    dvc_path = Path(repo_root) / dvc_file_path
    metadata = {}
    
    if not dvc_path.exists():
        return metadata
    
    try:
        with open(dvc_path, "r", encoding="utf-8") as f:
            dvc_data = yaml.safe_load(f)
        
        if dvc_data and "outs" in dvc_data and len(dvc_data["outs"]) > 0:
            out = dvc_data["outs"][0]
            metadata = {
                "dvc_md5_hash": out.get("md5", "unknown"),
                "dvc_nfiles": out.get("nfiles", 0),
                "dvc_size_bytes": out.get("size", 0),
            }
    except Exception as e:
        print(f"⚠️  Could not read DVC metadata: {e}")
    
    return metadata


def run_training(
    data_yaml=None,
    model_base=None,
    project=None,
    name=None,
    epochs=None,
    imgsz=None,
    batch=None,
    use_wandb=None,
):
    """Train YOLO and return the path to the best checkpoint."""

    runtime = get_settings().runtime
    data_yaml = data_yaml or runtime.stratified_yaml
    model_base = model_base or runtime.model_base
    project = project or runtime.train_project
    name = name or runtime.train_name
    epochs = epochs if epochs is not None else runtime.epochs
    imgsz = imgsz if imgsz is not None else runtime.imgsz
    batch = batch if batch is not None else runtime.batch
    use_wandb = use_wandb if use_wandb is not None else runtime.use_wandb

    # Try to initialize W&B — training continues even if it fails
    run = None
    dvc_meta = {}
    if use_wandb:
        try:
            wandb_key = os.getenv("WANDB_API_KEY")
            if wandb_key:
                wandb.login(key=wandb_key)

            # Load DVC metadata for reproducibility
            dvc_meta = _get_dvc_metadata(runtime.project_root)

            run = wandb.init(
                entity=runtime.wandb_entity,
                project=runtime.wandb_project,
                name=name,
                job_type="training",
                config={
                    "dataset_version": runtime.dataset_version,
                    "dataset_path": "dataset",
                    **dvc_meta,  # Include DVC hash, file count, size
                    "model": model_base,
                    "epochs": epochs,
                    "imgsz": imgsz,
                    "batch": batch,
                    "optimizer": runtime.optimizer,
                    "lr0": runtime.lr0,
                    "cls": runtime.cls,
                },
            )
            print("✅ W&B initialized successfully.")
            if dvc_meta:
                print(f"  Dataset version (DVC): {dvc_meta.get('dvc_md5_hash', 'unknown')}")
                print(f"  Dataset files: {dvc_meta.get('dvc_nfiles', '?')}")
        except Exception as e:
            print(f"⚠️  W&B login failed: {e}")
            print("⚠️  Continuing training without W&B logging...\n")

    resume_training = _env_bool("APP_TRAIN_RESUME", False)
    strict_resume = _env_bool("APP_TRAIN_STRICT_RESUME", False)
    amp = _env_bool("APP_TRAIN_AMP", True)
    last_checkpoint = Path(project) / name / "weights" / "last.pt"

    train_options = {}
    if resume_training and last_checkpoint.exists():
        model = YOLO(str(last_checkpoint))
        if strict_resume:
            print(f"Resuming training state from: {last_checkpoint}")
            train_options["resume"] = True
        else:
            print(f"Warm-starting training from: {last_checkpoint}")
    else:
        model = YOLO(model_base)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        conf=runtime.conf,
        iou=runtime.iou,
        cls=runtime.cls,
        augment=runtime.augment,
        optimizer=runtime.optimizer,
        lr0=runtime.lr0,
        lrf=runtime.lrf,
        warmup_epochs=runtime.warmup_epochs,
        patience=runtime.patience,
        device=runtime.device,
        workers=runtime.workers,
        project=project,
        name=name,
        exist_ok=True,
        amp=amp,
        plots=True,
        **train_options,
    )

    best = os.path.join(str(results.save_dir), "weights", "best.pt")
    print(f"\n── Training complete ──")
    print(f"Best model saved to: {best}")

    # Log to W&B only if the run was successfully initialized
    if run is not None:
        try:
            # Log final metrics to summary
            run.summary["best_precision"] = _metric_value(results, "metrics/precision(B)")
            run.summary["best_recall"] = _metric_value(results, "metrics/recall(B)")
            run.summary["best_mAP50"] = _metric_value(results, "metrics/mAP50(B)")
            run.summary["best_mAP50_95"] = _metric_value(results, "metrics/mAP50-95(B)")
            run.summary["best_fitness"] = _fitness_value(results)
            
            # Log dataset version and DVC metadata to summary for easy tracking
            run.summary["dataset_version"] = runtime.dataset_version
            for key, value in dvc_meta.items():
                run.summary[key] = value

            # Log metrics and DVC info to the run
            log_dict = {
                "dataset_version": runtime.dataset_version,
                "final_precision": _metric_value(results, "metrics/precision(B)"),
                "final_recall": _metric_value(results, "metrics/recall(B)"),
                "final_mAP50": _metric_value(results, "metrics/mAP50(B)"),
                "final_mAP50_95": _metric_value(results, "metrics/mAP50-95(B)"),
                "total_epochs_trained": len(results.epochs) if hasattr(results, 'epochs') else epochs,
            }
            log_dict.update(dvc_meta)
            run.log(log_dict)
            
            # Log W&B Artifact with reference to DVC-tracked dataset
            # This creates a professional link: W&B tracks versioning, DVC manages files
            try:
                artifact = wandb.Artifact(f'retail-dataset-v{runtime.dataset_version}', type='dataset')
                dataset_path = os.path.join(runtime.project_root, "dataset")
                artifact.add_reference(f'file://{os.path.abspath(dataset_path)}')
                run.log_artifact(artifact)
                print("✅ W&B Artifact logged with DVC dataset reference.")
            except Exception as artifact_err:
                print(f"ℹ️  Could not log W&B Artifact: {artifact_err}")
            
            run.finish()
            print("✅ W&B run finished and metrics logged.")
        except Exception as e:
            print(f"⚠️  W&B logging failed after training: {e}")
    else:
        print("ℹ️  W&B was not active — metrics not logged remotely.")
        print(f"ℹ️  Results are saved locally at: {results.save_dir}")

    return best


if __name__ == '__main__':
    run_training()
