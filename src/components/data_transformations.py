import sys
import pickle
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from src.components.feature_extraction import FeatureExtraction


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    processed_data_dir = os.path.join("artifacts", "data", "processed")
    train_csv = os.path.join("artifacts", "data", "processed", "processed_train.csv")
    test_csv = os.path.join("artifacts", "data", "processed", "processed_test.csv")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Creating data transformer object")

            numerical_columns = [
                "open",
                "high",
                "low",
                "volume",
                "year",
                "month",
                "day",
                "day_of_week",
                "week_of_year",
                "quarter",
                "is_month_start",
                "is_month_end",
                "is_quarter_start",
                "is_quarter_end",
                "open_EMA_5",
                "open_EMA_10",
                "open_EMA_20",
                "high_EMA_5",
                "high_EMA_10",
                "high_EMA_20",
                "low_EMA_5",
                "low_EMA_10",
                "low_EMA_20",
                "close_EMA_5",
                "close_EMA_10",
                "close_EMA_20",
            ]
            categorical_columns = ["Name"]

            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

            num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", encoder),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        try:
            feature_extraction = FeatureExtraction()
            feature_extraction.initiate_feature_extraction()

            logging.info("Reading processed train data")
            df_train = pd.read_csv(self.data_transformation_config.train_csv)
            df_test = pd.read_csv(self.data_transformation_config.test_csv)

            if "close" not in df_train.columns or "close" not in df_test.columns:
                raise ValueError("'close' column not found in train/test dataset")

            X_train = df_train.drop(columns=["close"])
            y_train = df_train["close"]

            X_test = df_test.drop(columns=["close"])
            y_test = df_test["close"]

            preprocessor = self.get_data_transformer_object()

            logging.info("Applying preprocessing to train data")
            X_train_preprocessed = preprocessor.fit_transform(X_train)
            logging.info("Applying preprocessing to test data")
            X_test_preprocessed = preprocessor.transform(
                X_test
            )  # ✅ only transform test

            # Save preprocessor
            os.makedirs(
                os.path.dirname(
                    self.data_transformation_config.preprocessor_obj_file_path
                ),
                exist_ok=True,
            )
            with open(
                self.data_transformation_config.preprocessor_obj_file_path, "wb"
            ) as f:
                pickle.dump(preprocessor, f)
            logging.info(
                f"Preprocessor saved at {self.data_transformation_config.preprocessor_obj_file_path}"
            )

            # Save processed arrays
            os.makedirs(
                self.data_transformation_config.processed_data_dir, exist_ok=True
            )

            np.save(
                os.path.join(
                    self.data_transformation_config.processed_data_dir,
                    "processed_train.npy",
                ),
                X_train_preprocessed,
            )
            np.save(
                os.path.join(
                    self.data_transformation_config.processed_data_dir,
                    "target_train.npy",
                ),
                y_train.values,
            )

            np.save(
                os.path.join(
                    self.data_transformation_config.processed_data_dir,
                    "processed_test.npy",
                ),
                X_test_preprocessed,
            )
            np.save(
                os.path.join(
                    self.data_transformation_config.processed_data_dir,
                    "target_test.npy",
                ),
                y_test.values,
            )

            logging.info("Processed train & test arrays saved as .npy")

            return (
                X_train_preprocessed,
                y_train.values,
                X_test_preprocessed,
                y_test.values,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_transformation = DataTransformation()
    X_train, y_train, X_test, y_test = (
        data_transformation.initiate_data_transformation()
    )
    logging.info("Data transformation completed successfully")
