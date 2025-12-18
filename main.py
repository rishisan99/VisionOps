# main.py

import sys
from src.logging.logger import logging
from src.exception.exception import CustomException

from src.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataTransformationConfig,
    DataSplittingConfig,
)

from src.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.pipeline.data_splitting_pipeline import DataSplittingPipeline


if __name__ == "__main__":
    try:
        # One run = one timestamp
        training_pipeline_config = TrainingPipelineConfig()
        logging.info("> TrainingPipelineConfig created")

        # Phase 2: Ingestion
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion_pipeline = DataIngestionPipeline(data_ingestion_config)

        data_ingestion_artifact = data_ingestion_pipeline.run()
        print(data_ingestion_artifact)
        print("")

        # Phase 3: Transformation
        data_transformation_config = DataTransformationConfig(training_pipeline_config, data_ingestion_config)
        data_transformation_pipeline = DataTransformationPipeline(data_ingestion_artifact, data_transformation_config)

        data_transformation_artifact = data_transformation_pipeline.run()
        print(data_transformation_artifact)
        print("")

        # Phase 4: Splitting
        data_splitting_config = DataSplittingConfig(training_pipeline_config, data_transformation_config)
        data_splitting_pipeline = DataSplittingPipeline(data_transformation_artifact, data_splitting_config)

        data_splitting_artifact = data_splitting_pipeline.run()
        print(data_splitting_artifact)
        print("")

        print("✅ Layer 1 ETL Complete (Ingestion → Transformation → Splitting).")

    except Exception as e:
        raise CustomException(e, sys)
