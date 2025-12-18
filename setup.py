# setup.py
import os
from pathlib import Path

from src.constants import training_pipeline as tp


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main() -> None:
    # Only stable dirs:
    # - data/_source (for external source dataset)
    # - Artifacts (root for all generated deliverables)
    # - logs
    ensure_dir(os.path.join(tp.DATA_DIR, tp.SOURCE_DIR_NAME))
    ensure_dir(tp.ARTIFACTS_ROOT_DIR)
    ensure_dir(tp.LOGS_DIR)

    print("✅ Directory skeleton ensured:")
    print(f" - {os.path.join(tp.DATA_DIR, tp.SOURCE_DIR_NAME)}")
    print(f" - {tp.ARTIFACTS_ROOT_DIR}")
    print(f" - {tp.LOGS_DIR}")
    print("\n📌 Put your PlantVillage dataset under:")
    print(f" - {os.path.join(tp.DATA_DIR, tp.SOURCE_DIR_NAME, tp.DATASET_DIR_NAME)}")


if __name__ == "__main__":
    main()
