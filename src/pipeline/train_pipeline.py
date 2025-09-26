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
            data_ingestion=DataIngestion()
            train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

            data_transformation=DataTransformation()
            train_array,test_array,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

            model_trainer=ModelTrainer()
            model_trainer.initiate_model_training(train_array,test_array)
            logging.info("Pipeline completed successfully")

        except Exception as e:
            logging.error("Error occurred in pipeline")
            raise CustomException(e, sys)
           