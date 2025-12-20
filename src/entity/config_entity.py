# src/entity/config_entity.py

import os
from datetime import datetime
from typing import Dict, List, Tuple

from src.constants import training_pipeline as tp


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _validate_split_ratios(train: float, val: float, test: float) -> None:
    total = train + val + test
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total} (train={train}, val={val}, test={test})"
        )


def _validate_target_size(target_size: Tuple[int, int]) -> None:
    if not isinstance(target_size, tuple) or len(target_size) != 2:
        raise ValueError(f"target_size must be tuple(int,int), got: {target_size}")
    h, w = target_size
    if not (isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0):
        raise ValueError(f"target_size must contain positive ints, got: {target_size}")


def _validate_class_targets(class_targets: Dict[str, int]) -> None:
    if not class_targets or not isinstance(class_targets, dict):
        raise ValueError("class_targets must be a non-empty dict {class_name: count}")
    for k, v in class_targets.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError(f"Invalid class name: {k!r}")
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"Target count for '{k}' must be positive int, got: {v}")


def _validate_allowed_ext(allowed_ext: List[str]) -> None:
    if not allowed_ext or not isinstance(allowed_ext, list):
        raise ValueError("allowed_ext must be a non-empty list like ['.jpg', '.png']")
    for ext in allowed_ext:
        if not isinstance(ext, str) or not ext.startswith("."):
            raise ValueError(f"Invalid extension: {ext!r}")


class TrainingPipelineConfig:
    """
    Creates:
      - Artifacts/<timestamp>/
      - logs/
      - data/_source/

    IMPORTANT:
      Only source data lives under data/_source/.
      Everything generated goes under Artifacts/<timestamp>/...
    """

    def __init__(self, timestamp=datetime.now()):
        ts = timestamp.strftime("%m_%d_%Y_%H_%M_%S")

        self.pipeline_name: str = tp.PIPELINE_NAME
        self.artifact_root_dir: str = tp.ARTIFACTS_ROOT_DIR
        self.artifact_dir: str = os.path.join(self.artifact_root_dir, ts)

        self.logs_dir: str = tp.LOGS_DIR
        self.timestamp: str = ts

        # stable source root: data/_source/plantvillage
        self.source_root_dir: str = os.path.join(tp.DATA_DIR, tp.SOURCE_DIR_NAME, tp.DATASET_DIR_NAME)

        # Ensure stable + run directories exist
        _ensure_dir(os.path.join(tp.DATA_DIR, tp.SOURCE_DIR_NAME))
        _ensure_dir(self.artifact_root_dir)
        _ensure_dir(self.artifact_dir)
        _ensure_dir(self.logs_dir)


class DataIngestionConfig:
    """
    Input:
      - data/_source/plantvillage/...

    Outputs (ALL under Artifacts/<ts>/data_ingestion/):
      - raw/
      - raw_bad/
      - metadata/*.json
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Behavior knobs (locked subset)
        self.class_targets: Dict[str, int] = {
            "Tomato_healthy": 1000,
            "Tomato_Early_blight": 1000,
            "Tomato_Late_blight": 1000,
            "Tomato_Septoria_leaf_spot": 1000,
        }
        self.seed: int = 42
        self.allowed_ext: List[str] = [".jpg", ".jpeg", ".png"]
        self.max_corrupt_rate: float = 0.01

        _validate_class_targets(self.class_targets)
        _validate_allowed_ext(self.allowed_ext)
        if not isinstance(self.seed, int):
            raise ValueError("seed must be int")
        if not (0.0 <= self.max_corrupt_rate <= 1.0):
            raise ValueError("max_corrupt_rate must be in [0,1]")

        # Inputs
        self.source_root_dir: str = training_pipeline_config.source_root_dir

        # Stage base dir
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            tp.DATA_INGESTION_DIR_NAME
        )

        # Outputs
        self.raw_data_dir: str = os.path.join(self.data_ingestion_dir, tp.INGEST_RAW_DIR_NAME)
        self.raw_bad_data_dir: str = os.path.join(self.data_ingestion_dir, tp.INGEST_RAW_BAD_DIR_NAME)
        self.metadata_dir: str = os.path.join(self.data_ingestion_dir, tp.INGEST_METADATA_DIR_NAME)

        _ensure_dir(self.data_ingestion_dir)
        _ensure_dir(self.raw_data_dir)
        _ensure_dir(self.raw_bad_data_dir)
        _ensure_dir(self.metadata_dir)

        # Metadata files
        self.class_map_file_path: str = os.path.join(self.metadata_dir, tp.CLASS_MAP_FILE)
        self.source_scan_file_path: str = os.path.join(self.metadata_dir, tp.SOURCE_SCAN_FILE)
        self.selected_files_file_path: str = os.path.join(self.metadata_dir, tp.SELECTED_FILES_FILE)
        self.ingest_report_file_path: str = os.path.join(self.metadata_dir, tp.INGEST_REPORT_FILE)
        self.dataset_stats_raw_file_path: str = os.path.join(self.metadata_dir, tp.DATASET_STATS_RAW_FILE)


class DataTransformationConfig:
    """
    Input:
      - Artifacts/<ts>/data_ingestion/raw

    Outputs (ALL under Artifacts/<ts>/data_transformation/):
      - processed/all
      - processed/preview
      - metadata/*.json
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig, data_ingestion_config: DataIngestionConfig):
        self.target_size: Tuple[int, int] = (224, 224)
        _validate_target_size(self.target_size)

        # Input
        self.input_raw_data_dir: str = data_ingestion_config.raw_data_dir

        # Stage base dir
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            tp.DATA_TRANSFORMATION_DIR_NAME
        )

        processed_root = os.path.join(self.data_transformation_dir, tp.TRANSFORM_PROCESSED_DIR_NAME)
        self.processed_all_dir: str = os.path.join(processed_root, tp.TRANSFORM_ALL_DIR_NAME)
        self.processed_preview_dir: str = os.path.join(processed_root, tp.TRANSFORM_PREVIEW_DIR_NAME)

        self.metadata_dir: str = os.path.join(self.data_transformation_dir, tp.TRANSFORM_METADATA_DIR_NAME)

        _ensure_dir(self.data_transformation_dir)
        _ensure_dir(processed_root)
        _ensure_dir(self.processed_all_dir)
        _ensure_dir(self.processed_preview_dir)
        _ensure_dir(self.metadata_dir)

        self.transform_report_file_path: str = os.path.join(self.metadata_dir, tp.TRANSFORM_REPORT_FILE)
        self.dataset_stats_processed_file_path: str = os.path.join(self.metadata_dir, tp.DATASET_STATS_PROCESSED_FILE)


class DataSplittingConfig:
    """
    Input:
      - Artifacts/<ts>/data_transformation/processed/all

    Outputs (ALL under Artifacts/<ts>/data_splitting/):
      - splits/splits.csv
      - processed_split/{train,val,test}/
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig, data_transformation_config: DataTransformationConfig):
        self.train_ratio: float = 0.70
        self.val_ratio: float = 0.15
        self.test_ratio: float = 0.15
        self.seed: int = 42

        _validate_split_ratios(self.train_ratio, self.val_ratio, self.test_ratio)
        if not isinstance(self.seed, int):
            raise ValueError("seed must be int")

        # Input
        self.input_processed_all_dir: str = data_transformation_config.processed_all_dir

        # Stage base dir
        self.data_splitting_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            tp.DATA_SPLITTING_DIR_NAME
        )

        self.splits_dir: str = os.path.join(self.data_splitting_dir, tp.SPLITS_DIR_NAME)
        self.splits_csv_path: str = os.path.join(self.splits_dir, tp.SPLITS_CSV_FILE)
        self.split_report_file_path: str = os.path.join(self.splits_dir, tp.SPLIT_REPORT_FILE)

        self.processed_split_dir: str = os.path.join(self.data_splitting_dir, tp.PROCESSED_SPLIT_DIR_NAME)
        self.train_dir: str = os.path.join(self.processed_split_dir, tp.TRAIN_DIR_NAME)
        self.val_dir: str = os.path.join(self.processed_split_dir, tp.VAL_DIR_NAME)
        self.test_dir: str = os.path.join(self.processed_split_dir, tp.TEST_DIR_NAME)

        _ensure_dir(self.data_splitting_dir)
        _ensure_dir(self.splits_dir)
        _ensure_dir(self.processed_split_dir)
        _ensure_dir(self.train_dir)
        _ensure_dir(self.val_dir)
        _ensure_dir(self.test_dir)


class ModelTrainingConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_training_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            tp.MODEL_TRAINING_DIR_NAME
        )

        os.makedirs(self.model_training_dir, exist_ok=True)

class RepresentationLearningConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.ssl_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            tp.MODEL_TRAINING_DIR_NAME,
            tp.SSL_REPRESENTATION_DIR_NAME
        )

        self.checkpoints_dir: str = os.path.join(self.ssl_dir, tp.CHECKPOINTS_DIR_NAME)
        self.metrics_dir: str = os.path.join(self.ssl_dir, tp.METRICS_DIR_NAME)

        # Training params (CPU/MPS friendly)
        self.epochs: int = 25 #10
        self.batch_size: int = 64
        self.learning_rate: float = 3e-4
        self.temperature: float = 0.5

        os.makedirs(self.ssl_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        self.ssl_metrics_file_path: str = os.path.join(
            self.metrics_dir, tp.SSL_METRICS_FILE
        )

class FineTuneModelConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.finetune_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            tp.MODEL_TRAINING_DIR_NAME,
            tp.FINETUNE_CLASSIFIER_DIR_NAME
        )

        self.checkpoints_dir: str = os.path.join(self.finetune_dir, tp.CHECKPOINTS_DIR_NAME)
        self.metrics_dir: str = os.path.join(self.finetune_dir, tp.METRICS_DIR_NAME)

        # Training params
        self.epochs: int = 25 #15
        self.batch_size: int = 32
        self.learning_rate: float = 1e-3
        self.freeze_backbone: bool = True

        os.makedirs(self.finetune_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        self.metrics_file_path: str = os.path.join(
            self.metrics_dir, tp.FINETUNE_METRICS_FILE
        )
        self.confusion_matrix_path: str = os.path.join(
            self.metrics_dir, tp.CONFUSION_MATRIX_FILE
        )
        
        # -------------------------
        # Two-phase fine-tuning (accuracy boost)
        # -------------------------
        self.phase1_epochs: int = 20          # head-only epochs (keep most epochs here)
        self.phase2_unfreeze_layer4: bool = True
        self.phase2_epochs: int = 5           # 3–5 is enough
        self.phase2_learning_rate: float = 1e-4  # small LR for stability on MPS

class ExplainabilityConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.explainability_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            tp.MODEL_TRAINING_DIR_NAME,
            tp.EXPLAINABILITY_DIR_NAME
        )

        self.heatmaps_dir: str = os.path.join(self.explainability_dir, "heatmaps")

        # How many samples to explain
        self.samples_per_class: int = 5

        os.makedirs(self.explainability_dir, exist_ok=True)
        os.makedirs(self.heatmaps_dir, exist_ok=True)

        self.index_file_path: str = os.path.join(
            self.explainability_dir, tp.EXPLAINABILITY_INDEX_FILE
        )

class MLflowConfig:
    def __init__(self):
        # Local default; later override with DagsHub URI
        self.tracking_uri: str | None = os.getenv("MLFLOW_TRACKING_URI", None)
        self.experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "VisionOps")
