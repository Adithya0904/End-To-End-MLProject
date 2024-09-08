import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings("ignore")

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}
        
        for model in models:
            m=models[model]
            m.fit(X_train,y_train)

            para=param[model]

            gs = RandomizedSearchCV(m,para,cv=3)
            gs.fit(X_train,y_train)

            m.set_params(**gs.best_params_)
            m.fit(X_train,y_train)

            prediction=m.predict(X_test)
            test_accuracy=r2_score(y_test,prediction)

            report[model]=test_accuracy

        return report
    except Exception as e:
        raise CustomException(e,sys)
