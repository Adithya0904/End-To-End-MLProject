import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate

@dataclass
class ModelTrainingConfig:
    trained_model_file_path=os.path.join('artifact','model.pkl')

class ModelTraining:
    def __init__(self) -> None:
        self.model_trainer_config=ModelTrainingConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info('X_train,y_train,X_test,y_test split')
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report=evaluate(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_test_accuracy=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_test_accuracy)]

            best_model=models[best_model_name]

            if best_test_accuracy<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset is {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
