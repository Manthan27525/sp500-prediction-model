import sys
from dataclasses import dataclass
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging


@dataclass
class FeatureExtractionConfig:
    raw_train_data_path = os.path.join("artifacts", "data", "train.csv")
    processed_train_data_path = os.path.join(
        "artifacts", "data", "processed", "processed_train.csv"
    )
    raw_test_data_path = os.path.join("artifacts", "data", "test.csv")
    processed_test_data_path = os.path.join(
        "artifacts", "data", "processed", "processed_test.csv"
    )


class FeatureExtraction:
    def __init__(self):
        self.feature_extraction_config = FeatureExtractionConfig()

    def initiate_feature_extraction(self):
        try:
            for i in [
                self.feature_extraction_config.raw_train_data_path,
                self.feature_extraction_config.raw_test_data_path,
            ]:
                if i == self.feature_extraction_config.raw_train_data_path:
                    self.feature_extraction_config.path = (
                        self.feature_extraction_config.raw_train_data_path
                    )
                    self.feature_extraction_config.processed_data_path = (
                        self.feature_extraction_config.processed_train_data_path
                    )
                else:
                    self.feature_extraction_config.path = (
                        self.feature_extraction_config.raw_test_data_path
                    )
                    self.feature_extraction_config.processed_data_path = (
                        self.feature_extraction_config.processed_test_data_path
                    )
                logging.info("Feature Extraction initiated")

                df = pd.read_csv(self.feature_extraction_config.path)
                logging.info("Raw data read successfully")
                df["date"] = pd.to_datetime(df["date"])
                df["year"] = df["date"].dt.year
                df["month"] = df["date"].dt.month
                df["day"] = df["date"].dt.day
                df["day_of_week"] = df["date"].dt.dayofweek
                df["week_of_year"] = df["date"].dt.isocalendar().week
                df["quarter"] = df["date"].dt.quarter
                df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
                df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
                df["is_quarter_start"] = df["date"].dt.is_quarter_start.astype(int)
                df["is_quarter_end"] = df["date"].dt.is_quarter_end.astype(int)
                logging.info("Feature 'Price_Range' added successfully")
                cols = ["open", "high", "low", "close"]
                windows = [5, 10, 20]
                for col in cols:
                    for w in windows:
                        df[f"{col}_EMA_{w}"] = df[col].ewm(span=w, adjust=False).mean()
                df.drop(columns=["date"], inplace=True)
                logging.info("Technical indicators added successfully")
                os.makedirs(
                    os.path.dirname(self.feature_extraction_config.processed_data_path),
                    exist_ok=True,
                )
                df.to_csv(
                    self.feature_extraction_config.processed_data_path, index=False
                )
                logging.info(
                    f"Processed data saved at {self.feature_extraction_config.processed_data_path}"
                )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    feature_extraction = FeatureExtraction()
    feature_extraction.initiate_feature_extraction()
