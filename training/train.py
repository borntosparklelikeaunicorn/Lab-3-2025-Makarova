# -*- coding: utf-8 -*-
from typing import List, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from clearml import Task, Dataset, Model
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

PROJECT = "Weather_Forecast_Project"
TRAIN_TASK_NAME = "Train_Model_Base"
DATASET_ALIAS = "Samara_Weather_History"
TARGET = "temperature_2m_mean"
VALIDATION_DAYS = 30

def enrich_time_series(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame.sort_values("time")
    frame["doy"] = frame["time"].dt.dayofyear
    frame["weekday"] = frame["time"].dt.weekday
    frame["month"] = frame["time"].dt.month
    frame["season_sin"] = np.sin(2 * np.pi * frame["doy"] / 365.25)
    frame["season_cos"] = np.cos(2 * np.pi * frame["doy"] / 365.25)
    for shift in (1, 2, 3, 7, 14):
        frame[f"lag_{shift}"] = frame[TARGET].shift(shift)
    for window in (7, 14, 30):
        shifted = frame[TARGET].shift(1)
        frame[f"mean_{window}"] = shifted.rolling(window).mean()
        frame[f"std_{window}"] = shifted.rolling(window).std()
    return frame.dropna().reset_index(drop=True)

def select_features(df: pd.DataFrame) -> List[str]:
    base = ["doy", "weekday", "month", "season_sin", "season_cos"]
    derived = [c for c in df.columns if c.startswith(("lag_", "mean_", "std_"))]
    return base + derived

def load_dataset() -> pd.DataFrame:
    dataset = Dataset.get(dataset_project=PROJECT, dataset_name=DATASET_ALIAS)
    path = dataset.get_local_copy()
    csv_name = next(f for f in os.listdir(path) if f.endswith(".csv"))
    return pd.read_csv(os.path.join(path, csv_name))

def train_val_split(df: pd.DataFrame, val_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return df.iloc[:-val_size], df.iloc[-val_size:]

def build_model(params: dict) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=params["iterations"],
        depth=params["depth"],
        learning_rate=params["learning_rate"],
        l2_leaf_reg=params["l2_leaf_reg"],
        loss_function=params["loss_function"],
        posterior_sampling=False,  # отключено для HPO
        random_seed=params["random_seed"],
        verbose=False,
    )

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }

def log_feature_importance(model: CatBoostRegressor, features: List[str], task: Task) -> None:
    importance = model.get_feature_importance()
    plt.figure(figsize=(10, 6))
    plt.barh(features, importance)
    plt.title("Feature Importance")
    plt.tight_layout()
    image_path = "feature_importance.png"
    plt.savefig(image_path)
    task.get_logger().report_image(title="Feature Importance", series="CatBoost", iteration=0, local_path=image_path)

def run_training_pipeline() -> None:
    task = Task.init(project_name=PROJECT, task_name=TRAIN_TASK_NAME, task_type=Task.TaskTypes.training)
    hyperparams = task.connect({
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.1,
        "l2_leaf_reg": 3.0,
        "loss_function": "RMSEWithUncertainty",
        "random_seed": 42,
    })

    raw_df = load_dataset()
    prepared_df = enrich_time_series(raw_df)
    train_df, val_df = train_val_split(prepared_df, VALIDATION_DAYS)
    features = select_features(prepared_df)
    X_train, y_train = train_df[features], train_df[TARGET]
    X_val, y_val = val_df[features], val_df[TARGET]

    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

    model = build_model(hyperparams)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

    raw_preds = model.predict(X_val)
    preds = raw_preds[:, 0] if raw_preds.ndim > 1 else raw_preds
    metrics = evaluate(y_val.values, preds)

    for name, value in metrics.items():
        task.get_logger().report_scalar(title="Metrics", series=name, value=value, iteration=0)

    print("Validation metrics:", ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    log_feature_importance(model, features, task)

    model_path = "catboost_model.cbm"
    model.save_model(model_path)

    task.upload_artifact(name="model", artifact_object=model_path)

    print("Training task finished successfully.")

if __name__ == "__main__":
    run_training_pipeline()
