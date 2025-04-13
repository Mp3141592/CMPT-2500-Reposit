#!/usr/bin/env python3

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',    handlers=[
        logging.StreamHandler()])  

logger = logging.getLogger(__name__)

"""
File should import the cleaned split data, encode the features, train using a ridge model with best params, and then export model to model folder.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error # Import necessary metrics
import matplotlib.pyplot as plt
import os
import shutil
import mlflow
import mlflow.sklearn 
from utils.arg_parser import get_input_args
from src.utils.monitoring import get_training_monitor

in_arg = get_input_args()

class Train:
    def __init__(self,X_train_path, X_test_path, y_train_path, y_test_path, solver, alpha, fit_intercept):
        # Paths to files

        self.X_train_path = X_train_path
        self.X_test_path = X_test_path
        self.y_train_path = y_train_path
        self.y_test_path = y_test_path
        self.solver = solver
        self.alpha = alpha 
        self.fit_intercept = fit_intercept

    def trainmodel(self):

        try: 
            
            monitor = get_training_monitor(port=8002)
            monitor.start()
            logger.info(f"Training commencing")


            # Start an MLflow run using the context manager
            with mlflow.start_run(run_name=f"GoAuto{in_arg.alpha}") as run:
                
                mlflow_tracking_uri = os.environ.get("http://localhost:5000")
                mlflow.set_tracking_uri(mlflow_tracking_uri)

                mlflow.sklearn.autolog()
                run_id = run.info.run_id

                # Log training parameters (done by autolog)
                """
                mlflow.log_param('solver', 'auto')
                mlflow.log_param("alpha", 0.1)
                mlflow.log_param("fit_intercept", True)
                """
                # Load data
                X_train = pd.read_csv(self.X_train_path)
                X_test = pd.read_csv(self.X_test_path)
                y_train = pd.read_csv(self.y_train_path)
                y_test = pd.read_csv(self.y_test_path)

                # Encode columns
                X_train = pd.get_dummies(X_train,prefix=['transmission_from_vin'], columns = ['transmission_from_vin'], drop_first=True, dtype=float)
                X_train = pd.get_dummies(X_train,prefix=['stock_type'], columns = ['stock_type'], drop_first=True, dtype=float)
                X_train = pd.get_dummies(X_train,prefix=['make'], columns = ['make'], drop_first=False, dtype=float)
                X_test = pd.get_dummies(X_test,prefix=['transmission_from_vin'], columns = ['transmission_from_vin'], drop_first=True, dtype=float)
                X_test = pd.get_dummies(X_test,prefix=['stock_type'], columns = ['stock_type'], drop_first=True, dtype=float)
                X_test = pd.get_dummies(X_test,prefix=['make'], columns = ['make'], drop_first=False, dtype=float)

                """
                Removed pipeline since it was messing with export of model and instead manually input best params
                """


                #X_train = X_train.drop(columns=['make_Suzuki'])
                #X_test = X_test.drop(columns=['make_Suzuki'])

                #Create actual model with best params found after applying scaler
                MMS = MaxAbsScaler()
                MMS.fit(X_train)



                model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept, solver=self.solver)
                model = model.fit(X_train, y_train)

                y_hat = model.predict(X_test)

                mse = mean_squared_error(y_test, y_hat)

                monitor.record_metrics(mse=mse)

                # Done by autolog
                #mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_train.iloc[:1])
                
                import joblib

                # Target folder to move model to
                target_folder = "/home/machine/cmpt3830/models"
                target_folder = "/app/models"

                # Save the model with joblib 
                joblib.dump(model , 'ridge_model_v2.jlib')

                # export model to models file
                destination_path = os.path.join(target_folder, "ridge_model_v2.jlib")
                shutil.move("ridge_model_v2.jlib", destination_path)

                # Trouble shooting below
                autolog_run = mlflow.active_run()

                
                print(f"Autologging Nested Run ID: {autolog_run.info.run_id}")
                print(f"Run ID: {run.info.run_id}")

                logger.info("Training finished")
                return run_id
        
        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            raise