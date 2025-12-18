# src/utils/artifact_builder.py
from __future__ import annotations

from pathlib import Path

from src.constants.filenames import (
    FILE_CLASS_MAP,
    FILE_SOURCE_SCAN,
    FILE_SELECTED_FILES,
    FILE_INGEST_REPORT,
    FILE_DATASET_STATS_RAW,
    FILE_TRANSFORM_REPORT,
    FILE_DATASET_STATS_PROCESSED,
    FILE_SPLITS_CSV,
)
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataSplittingArtifact,
    ETLArtifacts,
)
from src.entity.config_entity import AppConfig
from src.exception.exception import CustomException
from src.logging.logger import logging


def ensure_dirs_exist(*dirs: Path) -> None:
    try:
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Failed creating directories: {e}")
        raise CustomException(e)


def build_etl_artifacts(cfg: AppConfig) -> ETLArtifacts:
    """
    Build strongly-typed artifact paths based on config.
    Does not create files. Can create required directories.
    """
    try:
        # Core dirs
        raw_dir = cfg.paths.raw_dir
        bad_dir = cfg.paths.bad_dir
        processed_dir = cfg.paths.processed_dir
        splits_dir = cfg.paths.splits_dir
        meta_dir = cfg.paths.metadata_dir

        processed_all = processed_dir / "all"
        processed_preview = processed_dir / "preview"

        processed_train = processed_dir / "train"
        processed_val = processed_dir / "val"
        processed_test = processed_dir / "test"

        # Ensure directory skeleton exists (Phase 1 requirement)
        ensure_dirs_exist(
            raw_dir, bad_dir,
            processed_dir, processed_all, processed_preview,
            splits_dir, meta_dir,
            processed_train, processed_val, processed_test
        )

        ingestion = DataIngestionArtifact(
            raw_data_dir=raw_dir,
            bad_data_dir=bad_dir,
            selected_files_json=meta_dir / FILE_SELECTED_FILES,
            ingest_report_json=meta_dir / FILE_INGEST_REPORT,
            dataset_stats_raw_json=meta_dir / FILE_DATASET_STATS_RAW,
            source_scan_json=meta_dir / FILE_SOURCE_SCAN,
            class_map_json=meta_dir / FILE_CLASS_MAP,
        )

        transformation = DataTransformationArtifact(
            processed_all_dir=processed_all,
            processed_preview_dir=processed_preview,
            transform_report_json=meta_dir / FILE_TRANSFORM_REPORT,
            dataset_stats_processed_json=meta_dir / FILE_DATASET_STATS_PROCESSED,
        )

        splitting = DataSplittingArtifact(
            splits_csv=splits_dir / FILE_SPLITS_CSV,
            processed_train_dir=processed_train,
            processed_val_dir=processed_val,
            processed_test_dir=processed_test,
        )

        return ETLArtifacts(
            ingestion=ingestion,
            transformation=transformation,
            splitting=splitting,
        )

    except Exception as e:
        raise CustomException(e)
