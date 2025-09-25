import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging



def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys) from e
    
def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]

            model.set_params(**para)
            # model.set_params(para)
            model.fit(X_train,y_train)

            y_test_pred=model.predict(X_test)

            r2_square=r2_score(y_test,y_test_pred)
            mae=mean_absolute_error(y_test,y_test_pred)
            rmse=np.sqrt(mean_squared_error(y_test,y_test_pred))

            report[list(models.keys())[i]]=r2_square

        return report
    except Exception as e:
        raise CustomException(e,sys) from e
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys) from e