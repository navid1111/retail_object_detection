"""Rebuild train/validation/test splits from the repository root."""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    os.environ["APP_PROJECT_ROOT"] = str(repo_root)
    os.environ["APP_STRATIFIED_YAML"] = "dataset/dataset_stratified/data.yaml"

    from src.data.rebuild_splits import main as rebuild_main

    rebuild_main(base_dir=repo_root)


if __name__ == "__main__":
    main()
