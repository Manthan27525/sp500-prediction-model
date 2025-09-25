import sys
from dataclasses import dataclass
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline   
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from src.components.feature_extraction import FeatureExtraction

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            logging.info("Data Transformation initiated")
            numerical_columns=['Open', 'High', 'Low', 'Close', 'Volume']
            categorical_columns=['Name']

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='mean')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',LabelEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
   # Data_transformation.py

    def initiate_data_transformation(self):
        try:
            feature_extraction=FeatureExtraction()
            feature_extraction.initiate_feature_extraction()
            logging.info("Obtaining preprocessing object")
            preprocessor_obj=self.get_data_transformer_object()
            
            df=pd.read_csv(feature_extraction.feature_extraction_config.processed_data_path)
            logging.info("Read processed data successfully")

            target_column_name='close'
            input_feature_names=df.drop(columns=[target_column_name]).columns
            
            X=df[input_feature_names]
            y=df[target_column_name]

            X_preprocessed=preprocessor_obj.fit_transform(X)
            logging.info("Applied preprocessing object on training dataframe and testing dataframe.")

            return X_preprocessed, y, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e,sys)