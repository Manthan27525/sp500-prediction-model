import os
import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression ,Ridge , Lasso, ElasticNet    
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from artifacts.best_params import best_params
from src.logger import logging
from skllearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.utils import save_object, evaluate_models
from dataclasses import dataclass
import sys

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    scaler_path=os.path.join("artifacts","scaler.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            scaler=StandardScaler()
            X_train=scaler.fit_transform(X_train)
            X_test=scaler.transform(X_test)

            models={
                "Linear Regression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "ElasticNet":ElasticNet(),
                "KNeighbors Regressor":KNeighborsRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Random Forest":RandomForestRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "XGBRegressor":XGBRegressor(),
                "LGBMRegressor":LGBMRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor(),
                "Bagging Regressor":BaggingRegressor(),
                "SVR":SVR()
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=best_params)

            ## To get the best model score from the dictionary
            best_model_score=max(sorted(model_report.values()))

            ## To get the best model name from the dictionary
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            save_object(
                file_path=self.model_trainer_config.scaler_path,
                obj=scaler
            )

            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            mae=mean_absolute_error(y_test,predicted)
            rmse=np.sqrt(mean_squared_error(y_test,predicted))

            return r2_square,mae,rmse

        except Exception as e:
            raise CustomException(e,sys) from e