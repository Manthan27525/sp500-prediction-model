import os
import json
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import sys

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    scaler_path = os.path.join("artifacts", "scaler.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Scaling features")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Candidate models
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "LGBMRegressor": LGBMRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Bagging Regressor": BaggingRegressor(),
                "SVR": SVR(),
            }

            # Load best hyperparameters
            best_params_path = "artifacts/best_params/best_params.json"

            with open(best_params_path, "r") as f:
                best_params = json.load(f)

            # Evaluate models
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=best_params,
            )

            # Pick best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found", sys)

            logging.info(f"Best model: {best_model_name} with R2: {best_model_score}")

            # Refit best model with best params
            best_model = models[best_model_name].set_params(
                **best_params.get(best_model_name, {})
            )
            best_model.fit(X_train, y_train)

            # Save model + scaler
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            save_object(file_path=self.model_trainer_config.scaler_path, obj=scaler)

            # Final evaluation
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            mae = mean_absolute_error(y_test, predicted)
            rmse = np.sqrt(mean_squared_error(y_test, predicted))

            return r2_square, mae, rmse

        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    logging.info("Model Trainer started")

    # Load processed data
    X_train = np.load("artifacts/data/processed/processed_train.npy", allow_pickle=True)
    y_train = np.load("artifacts/data/processed/target_train.npy", allow_pickle=True)

    X_test = np.load("artifacts/data/processed/processed_test.npy", allow_pickle=True)
    y_test = np.load("artifacts/data/processed/target_test.npy", allow_pickle=True)

    model_trainer = ModelTrainer()
    r2_square, mae, rmse = model_trainer.initiate_model_trainer(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    print(f"R2 Square: {r2_square:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    logging.info(f"R2 Square: {r2_square}")
    logging.info(f"MAE: {mae}")
    logging.info(f"RMSE: {rmse}")

    print("Execution completed")
