import sys
from dataclasses import dataclass
import os
import pandas as pd 
from src.exception import CustomException
from src.logger import logging

@dataclass
class FeatureExtractionConfig:
    raw_data_path=os.path.join('artifacts','data','sp500.csv')
    processed_data_path=os.path.join('artifacts','data','processed_sp500.csv')
    
class FeatureExtraction:
    def __init__(self):
        self.feature_extraction_config=FeatureExtractionConfig()
    
    def initiate_feature_extraction(self):
        try:
            logging.info("Feature Extraction initiated")
            df=pd.read_csv(self.feature_extraction_config.raw_data_path)
            logging.info("Raw data read successfully")

            # Example feature extraction: Adding a new feature 'Price_Range'
            df['date'] = pd.to_datetime(df['date'])

            # Day, month, year
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day

            # Day of the week (0 = Monday, 6 = Sunday)
            df['day_of_week'] = df['date'].dt.dayofweek

            # Week of the year
            df['week_of_year'] = df['date'].dt.isocalendar().week

            # Quarter of the year
            df['quarter'] = df['date'].dt.quarter

            # Is month start/end
            df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
            df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

            # Is quarter start/end
            df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
            df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
            logging.info("Feature 'Price_Range' added successfully")
            
            cols = ['open', 'high', 'low', 'close']
            windows = [5, 10, 20]

            for col in cols:
                for w in windows:
                    df[f'{col}_EMA_{w}'] = df[col].ewm(span=w, adjust=False).mean()
                    
            df.drop(columns=['date'], inplace=True)
            logging.info("Technical indicators added successfully")

            # Save the processed data
            os.makedirs(os.path.dirname(self.feature_extraction_config.processed_data_path), exist_ok=True)
            df.to_csv(self.feature_extraction_config.processed_data_path, index=False)
            logging.info(f"Processed data saved at {self.feature_extraction_config.processed_data_path}")

            return self.feature_extraction_config.processed_data_path
        except Exception as e:
            raise CustomException(e,sys)
