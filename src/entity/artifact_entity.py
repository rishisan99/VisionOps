# src/entity/artifact_entity.py

from dataclasses import dataclass
from typing import Dict

@dataclass
class DataIngestionArtifact:
    raw_data_dir: str
    raw_bad_data_dir: str
    metadata_dir: str

    class_map_file_path: str
    source_scan_file_path: str
    selected_files_file_path: str
    ingest_report_file_path: str
    dataset_stats_raw_file_path: str

    # quick summary for printing
    total_selected: int
    total_copied: int
    total_corrupted: int
    per_class_selected: Dict[str, int]
    per_class_copied: Dict[str, int]
    per_class_corrupted: Dict[str, int]

@dataclass
class DataTransformationArtifact:
    processed_all_dir: str
    processed_preview_dir: str
    metadata_dir: str

    transform_report_file_path: str
    dataset_stats_processed_file_path: str

    total_input: int
    total_processed: int
    total_failed: int
    per_class_processed: Dict[str, int]
    per_class_failed: Dict[str, int]

@dataclass
class DataSplittingArtifact:
    splits_csv_path: str
    split_report_file_path: str

    processed_split_dir: str
    train_dir: str
    val_dir: str
    test_dir: str

    total_images: int
    train_count: int
    val_count: int
    test_count: int

    per_class_train: Dict[str, int]
    per_class_val: Dict[str, int]
    per_class_test: Dict[str, int]
    
# =========================
# Layer 2 Artifacts
# =========================

@dataclass
class RepresentationLearningArtifact:
    ssl_dir: str
    checkpoints_dir: str
    metrics_dir: str

    best_backbone_path: str
    ssl_metrics_file_path: str

    total_epochs: int
    final_loss: float


@dataclass
class ModelTrainerArtifact:
    finetune_dir: str
    checkpoints_dir: str
    metrics_dir: str

    trained_model_path: str
    metrics_file_path: str
    confusion_matrix_path: str

    accuracy: float
    f1_score: float
    precision: float
    recall: float


@dataclass
class ExplainabilityArtifact:
    explainability_dir: str
    heatmaps_dir: str
    index_file_path: str

    total_samples: int
