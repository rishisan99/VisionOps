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

from src.entity.config_entity import (
    ModelTrainingConfig,
    RepresentationLearningConfig,
    FineTuneModelConfig,
    ExplainabilityConfig
)

from src.entity.config_entity import RepresentationLearningConfig
from src.pipeline.representation_learning_pipeline import RepresentationLearningPipeline

from src.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.pipeline.data_splitting_pipeline import DataSplittingPipeline

from src.entity.config_entity import FineTuneModelConfig
from src.pipeline.model_trainer_pipeline import ModelTrainerPipeline

from src.entity.config_entity import ExplainabilityConfig
from src.pipeline.explainability_pipeline import ExplainabilityPipeline


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


        model_training_config = ModelTrainingConfig(training_pipeline_config)
        ssl_config = RepresentationLearningConfig(training_pipeline_config)
        finetune_config = FineTuneModelConfig(training_pipeline_config)
        explain_config = ExplainabilityConfig(training_pipeline_config)

        print("✅ Layer 2 configs initialized successfully")
        
        # Phase 5C: SSL Representation Learning
        ssl_config = RepresentationLearningConfig(training_pipeline_config)
        ssl_pipeline = RepresentationLearningPipeline(ssl_config)

        # Use train split dir as unlabeled pool
        ssl_train_dir = data_splitting_artifact.train_dir

        ssl_artifact = ssl_pipeline.run(ssl_train_dir=ssl_train_dir)
        print(ssl_artifact)
        print("\n✅ Phase 5C (Representation Learning) complete.")

        # Phase 6: Supervised Fine-tuning
        finetune_config = FineTuneModelConfig(training_pipeline_config)
        finetune_pipeline = ModelTrainerPipeline(finetune_config)

        finetune_artifact = finetune_pipeline.run(
            data_splitting_artifact=data_splitting_artifact,
            ssl_artifact=ssl_artifact
        )

        print(finetune_artifact)
        print("\n✅ Phase 6 (Fine-tuning) complete.")

        # Phase 7: Grad-CAM Explainability
        explain_config = ExplainabilityConfig(training_pipeline_config)
        explain_pipeline = ExplainabilityPipeline(explain_config)

        explain_artifact = explain_pipeline.run(
            data_splitting_artifact=data_splitting_artifact,
            finetune_artifact=finetune_artifact
        )

        print(explain_artifact)
        print("\n✅ Phase 7 (Grad-CAM Explainability) complete.")


    except Exception as e:
        raise CustomException(e, sys)
