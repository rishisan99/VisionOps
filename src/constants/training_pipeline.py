# src/constants/training_pipeline.py

# Pipeline meta
PIPELINE_NAME: str = "vision_etl_pipeline"
ARTIFACTS_ROOT_DIR: str = "Artifacts"
LOGS_DIR: str = "logs"

# Stable source data (ONLY this stays in /data)
DATA_DIR: str = "data"
SOURCE_DIR_NAME: str = "_source"
DATASET_DIR_NAME: str = "PlantVillage"  # keep stable; you will place dataset here

# Stage directory names (under Artifacts/<timestamp>/)
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_SPLITTING_DIR_NAME: str = "data_splitting"

# Ingestion outputs (under Artifacts/<ts>/data_ingestion/)
INGEST_RAW_DIR_NAME: str = "raw"
INGEST_RAW_BAD_DIR_NAME: str = "raw_bad"
INGEST_METADATA_DIR_NAME: str = "metadata"

# Transformation outputs (under Artifacts/<ts>/data_transformation/)
TRANSFORM_PROCESSED_DIR_NAME: str = "processed"
TRANSFORM_ALL_DIR_NAME: str = "all"
TRANSFORM_PREVIEW_DIR_NAME: str = "preview"
TRANSFORM_METADATA_DIR_NAME: str = "metadata"

# Splitting outputs (under Artifacts/<ts>/data_splitting/)
SPLITS_DIR_NAME: str = "splits"
PROCESSED_SPLIT_DIR_NAME: str = "processed_split"
TRAIN_DIR_NAME: str = "train"
VAL_DIR_NAME: str = "val"
TEST_DIR_NAME: str = "test"


# Filenames (reports + metadata)
CLASS_MAP_FILE: str = "class_map.json"
SOURCE_SCAN_FILE: str = "source_scan.json"
SELECTED_FILES_FILE: str = "selected_files.json"
INGEST_REPORT_FILE: str = "ingest_report.json"
DATASET_STATS_RAW_FILE: str = "dataset_stats_raw.json"

TRANSFORM_REPORT_FILE: str = "transform_report.json"
SPLIT_REPORT_FILE: str = "split_report.json"
DATASET_STATS_PROCESSED_FILE: str = "dataset_stats_processed.json"

SPLITS_CSV_FILE: str = "splits.csv"
