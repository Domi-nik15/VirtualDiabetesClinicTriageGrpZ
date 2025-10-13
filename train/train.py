import mlflow
import mlflow.sklearn



import os

import pandas as pd

import mlflow
from mlflow.models import infer_signature

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error

def read_dataframe():
  df = (pd.DataFrame) (load_diabetes(as_frame=True).frame)
  return df

def run():
  MODEL_NAME  = os.getenv("MODEL_NAME", "diabetes-progression")
  VERSION = os.getenv("MODEL_VERSION", "0.0")

  RANDOM_SEED = (int) (os.getenv("RANDOM_SEED", 0))
  TEST_SIZE = (float) (os.getenv("TEST_SIZE", 0.3))
  TARGET_VARIABLE = os.getenv("TARGET_VARIABLE", "target")
  SCALOR = "StandardScaler"
  MODEL = "LinearRegression"

  df = read_dataframe()
  
  X = df.drop(columns=TARGET_VARIABLE)
  y = df[TARGET_VARIABLE]

  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED) 

  with mlflow.start_run():
    mlflow.log_params({
    'Version': VERSION,
    'Scalor': SCALOR,
    'Model': MODEL,
    'TestPartionSize': TEST_SIZE,
    'TargetVariable' : TARGET_VARIABLE,
    'RandomSeed': RANDOM_SEED
    })

    pipeline = make_pipeline(
      StandardScaler(),
      LinearRegression()
    )
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    signature = infer_signature(X_val, y_pred)

    rmse = mean_squared_error(y_val, y_pred)
    print(f'RMSE on validation is {rmse}')
    mlflow.sklearn.save_model(sk_model=pipeline, 
                            path="artifacts/"+MODEL_NAME, 
                            signature=signature,
                            serialization_format="pickle"
                            )

if __name__ == '__main__':
    run()