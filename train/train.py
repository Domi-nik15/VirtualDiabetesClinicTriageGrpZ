import mlflow
import mlflow.sklearn

import os

import numpy as np
import pandas as pd

import mlflow
from mlflow.models import infer_signature

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, precision_score, recall_score


def read_dataframe():
    df = (pd.DataFrame)(load_diabetes(as_frame=True).frame)
    return df


def save_metrics(metrics: dict) -> None:
    header = "Metrics:"
    path = "artifacts/metrics.txt"
    save_file(header, path, metrics)


def save_parameters(parameters: dict) -> None:
    header = "Parameters:"
    path = "artifacts/parameters.txt"
    save_file(header, path, parameters)


def save_file(header: str, filepath: str, entries: dict) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as file:
        file.write(header + "\n\n")
        for key, value in entries.items():
            file.write(f"{key}: {value}\n")


def run():
    MODEL_NAME = os.getenv("MODEL_NAME", "diabetes-progression")

    RANDOM_SEED = (int)(os.getenv("RANDOM_SEED", 0))
    TEST_SIZE = (float)(os.getenv("TEST_SIZE", 0.3))
    TARGET_VARIABLE = os.getenv("TARGET_VARIABLE", "target")
    SCALOR = "StandardScaler"
    MODEL_TYPE = os.getenv("MODEL_TYPE", "linearReg")  # linearReg, ridge, randomForestReg
    CALIBRATION_THRESHOLD = float(os.getenv("RISK_THRESHOLD", 0.6))

    df = read_dataframe()

    X = df.drop(columns=TARGET_VARIABLE)
    y = df[TARGET_VARIABLE]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    with mlflow.start_run():
        parameters = {
            'Scalor': SCALOR,
            'Model': MODEL_TYPE,
            'TestPartionSize': TEST_SIZE,
            'TargetVariable': TARGET_VARIABLE,
            'RandomSeed': RANDOM_SEED
        }

        save_parameters(parameters)
        mlflow.log_params(parameters)

        if MODEL_TYPE == "linearReg":
            estimator = LinearRegression()
        elif MODEL_TYPE == "ridge":
            estimator = Ridge(random_state=RANDOM_SEED)
        elif MODEL_TYPE == "randomForestReg":
            estimator = RandomForestRegressor(random_state=RANDOM_SEED)
        else:
            raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")

        pipeline = make_pipeline(
            StandardScaler(),
            estimator
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        signature = infer_signature(X_val, y_pred)

        rmse = mean_squared_error(y_true=y_val, y_pred=y_pred)

        highrisk = np.quantile(y_val, CALIBRATION_THRESHOLD)
        y_val_highrisk = (y_val > highrisk).astype(int)
        y_pred_highrisk = (y_pred > highrisk).astype(int)

        precision = precision_score(y_true=y_val_highrisk, y_pred=y_pred_highrisk)
        recall = recall_score(y_true=y_val_highrisk, y_pred=y_pred_highrisk)

        metrics = {
            "RMSE": rmse,
            "High Risk Threshold": highrisk,
            "Precision": precision,
            "Recall": recall,
        }

        save_metrics(metrics=metrics)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.sklearn.save_model(sk_model=pipeline,
                                  path="artifacts/"+MODEL_NAME,
                                  signature=signature,
                                  serialization_format="pickle")


if __name__ == '__main__':
    run()
