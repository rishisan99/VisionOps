# src/pipeline/explainability_pipeline.py

import sys
from src.logging.logger import logging
from src.exception.exception import CustomException

from src.entity.config_entity import ExplainabilityConfig
from src.entity.artifact_entity import DataSplittingArtifact, ModelTrainerArtifact, ExplainabilityArtifact
from src.components.explainability import Explainability


class ExplainabilityPipeline:
    def __init__(self, explainability_config: ExplainabilityConfig):
        self.explainability_config = explainability_config

    def run(
        self,
        data_splitting_artifact: DataSplittingArtifact,
        finetune_artifact: ModelTrainerArtifact,
    ) -> ExplainabilityArtifact:
        try:
            logging.info("> Initiate Explainability (Grad-CAM)")
            explainer = Explainability(self.explainability_config)
            artifact = explainer.initiate_explainability(
                data_splitting_artifact=data_splitting_artifact,
                finetune_artifact=finetune_artifact
            )
            logging.info("> Explainability Completed")
            return artifact
        except Exception as e:
            raise CustomException(e, sys)
