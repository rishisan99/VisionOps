# src/pipeline/model_trainer_pipeline.py

import sys

from src.logging.logger import logging
from src.exception.exception import CustomException

from src.entity.config_entity import FineTuneModelConfig
from src.entity.artifact_entity import DataSplittingArtifact, RepresentationLearningArtifact, ModelTrainerArtifact
from src.components.model_trainer import FineTuneTrainer


class ModelTrainerPipeline:
    def __init__(self, finetune_config: FineTuneModelConfig):
        self.finetune_config = finetune_config

    def run(
        self,
        data_splitting_artifact: DataSplittingArtifact,
        ssl_artifact: RepresentationLearningArtifact,
    ) -> ModelTrainerArtifact:
        try:
            logging.info("> Initiate Supervised Fine-tuning (Model Trainer)")
            trainer = FineTuneTrainer(self.finetune_config, data_splitting_artifact, ssl_artifact)
            artifact = trainer.initiate_model_trainer()
            logging.info("> Supervised Fine-tuning Completed")
            return artifact
        except Exception as e:
            raise CustomException(e, sys)
