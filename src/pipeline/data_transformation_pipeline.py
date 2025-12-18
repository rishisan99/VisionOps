# src/pipeline/data_transformation_pipeline.py

import sys

from src.logging.logger import logging
from src.exception.exception import CustomException

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from src.components.data_transformation import DataTransformation


class DataTransformationPipeline:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_transformation_config: DataTransformationConfig):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_config = data_transformation_config

    def run(self) -> DataTransformationArtifact:
        try:
            logging.info("> Initiate the data transformation")
            transformation = DataTransformation(
                data_ingestion_artifact=self.data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config
            )
            artifact = transformation.initiate_data_transformation()
            logging.info("> Data Transformation Completed")
            return artifact

        except Exception as e:
            raise CustomException(e, sys)
