# src/pipeline/data_ingestion_pipeline.py

import sys
from src.logging.logger import logging
from src.exception.exception import CustomException

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.components.data_ingestion import DataIngestion


class DataIngestionPipeline:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def run(self) -> DataIngestionArtifact:
        try:
            logging.info("> Initiate the data ingestion")
            ingestion = DataIngestion(self.data_ingestion_config)
            artifact = ingestion.initiate_data_ingestion()
            logging.info("> Data Ingestion Completed")
            return artifact
        except Exception as e:
            raise CustomException(e, sys)
