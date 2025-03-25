#!/usr/bin/env python3

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
File should run through metrics to calculate how well the model is performing against test data.
"""
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error # Import necessary metrics
import joblib
import pandas as pd
import numpy as np
import mlflow.sklearn

class Eval:
    def __init__(self, y_test_path, X_test_path, model_path, run_id):
        self.X_test_path = X_test_path
        self.y_test_path = y_test_path
        self.model_path = model_path
        self.run_id = run_id
        
    def Evalulate(self):
        
        try:

            logger.info('Evaluating Training model with test data')
            # Load test datasets
            X_test = pd.read_csv(self.X_test_path)
            y_test = pd.read_csv(self.y_test_path)
            
            # Enocde test since original test file did not save new encoded data
            X_test = pd.get_dummies(X_test,prefix=['transmission_from_vin'], columns = ['transmission_from_vin'], drop_first=True, dtype=float)
            X_test = pd.get_dummies(X_test,prefix=['stock_type'], columns = ['stock_type'], drop_first=True, dtype=float)
            X_test = pd.get_dummies(X_test,prefix=['make'], columns = ['make'], drop_first=False, dtype=float)

            # Load logged model from mlflow
            model_uri = f"runs:/{self.run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)

            # Load the model directly
            # model = joblib.load(self.model_path)

            # Create y_hat_test
            y_hat_test = model.predict(X_test)

            # Create r2 and print 
            r2 = r2_score(y_test, y_hat_test)
            print(f"r2 score: {r2}")

            logger.info(f"Model evaluated to have an r2 score of {r2}")
            # mlflow autolog still runs from train py so commented out
            #mlflow.log_metric("r2_score", r2)

            logger.info("Evaluation finished")
            mlflow.end_run()

        except Exception as e:
            logger.error(f"Evaluation failed with error: {str(e)}")
            raise
            
