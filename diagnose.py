"""Run dataset diagnostics from the repository root."""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    os.environ["APP_PROJECT_ROOT"] = str(repo_root)

    from src.config import get_settings
    from src.data.diagnose import analyze_dataset

    runtime = get_settings().runtime
    print(f"Diagnosing stratified dataset: {runtime.stratified_yaml}")
    results = analyze_dataset(runtime.stratified_yaml)

    print("\nSummary")
    for split, stats in results.items():
        print(
            f"{split}: {stats['num_classes']} classes | "
            f"imbalance {stats['imbalance_ratio']}x | "
            f"{stats['total_objects']} objects"
        )


if __name__ == "__main__":
    main()
