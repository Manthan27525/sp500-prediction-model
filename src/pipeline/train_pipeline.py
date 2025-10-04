from src.components.data_ingestion import DataIngestion
from src.components.data_transformations import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
import sys
from src.logger import logging


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Pipeline started")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

            data_transformation = DataTransformation()
            X_train, y_train, X_test, y_test = (
                data_transformation.initiate_data_transformation()
            )

            model_trainer = ModelTrainer()
            r2_square, mae, rmse = model_trainer.initiate_model_trainer(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
            )

        except Exception as e:
            logging.error("Error occurred in pipeline")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = TrainPipeline()
    obj.run_pipeline()
