# src/pipeline/representation_learning_pipeline.py

import sys

from src.logging.logger import logging
from src.exception.exception import CustomException

from src.entity.config_entity import RepresentationLearningConfig
from src.entity.artifact_entity import RepresentationLearningArtifact
from src.components.representation_learning import RepresentationLearning


class RepresentationLearningPipeline:
    def __init__(self, representation_learning_config: RepresentationLearningConfig):
        self.representation_learning_config = representation_learning_config

    def run(self, ssl_train_dir: str) -> RepresentationLearningArtifact:
        try:
            logging.info("> Initiate Representation Learning (SSL)")
            trainer = RepresentationLearning(self.representation_learning_config)
            artifact = trainer.initiate_representation_learning(ssl_train_dir=ssl_train_dir)
            logging.info("> Representation Learning Completed")
            return artifact
        except Exception as e:
            raise CustomException(e, sys)
