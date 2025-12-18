# src/pipeline/data_splitting_pipeline.py

import sys

from src.logging.logger import logging
from src.exception.exception import CustomException

from src.entity.config_entity import DataSplittingConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataSplittingArtifact
from src.components.data_splitting import DataSplitting


class DataSplittingPipeline:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, data_splitting_config: DataSplittingConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.data_splitting_config = data_splitting_config

    def run(self) -> DataSplittingArtifact:
        try:
            logging.info("> Initiate the data splitting")
            splitting = DataSplitting(self.data_transformation_artifact, self.data_splitting_config)
            artifact = splitting.initiate_data_splitting()
            logging.info("> Data Splitting Completed")
            return artifact
        except Exception as e:
            raise CustomException(e, sys)
