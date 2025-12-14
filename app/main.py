from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import os

from catboost import CatBoostRegressor
from clearml import Model

APP_TITLE = "Weather ML Inference API"
CLEARML_PROJECT = "Weather_Forecast_Project"
CLEARML_MODEL_NAME = "weather_predictor_model"

DEFAULT_CITY = "Samara"
CITY_GEO = {
    "Samara": {"lat": 53.2001, "lon": 50.15}
}

FALLBACK_MODEL_PATH = "catboost_model.cbm"

class ForecastRequest(BaseModel):
    city: str
    dates: List[str]


class DailyForecast(BaseModel):
    date: str
    temperature: float
    uncertainty: float
    lower_bound: float
    upper_bound: float


class ForecastResponse(BaseModel):
    city: str
    predictions: List[DailyForecast]

class ModelLoader:
    def __init__(self):
        self._model = None

    def load(self) -> CatBoostRegressor:
        if self._model is not None:
            return self._model

        try:
            clearml_model = Model(
                project_name=CLEARML_PROJECT,
                model_name=CLEARML_MODEL_NAME
            )
            model_path = clearml_model.get_local_copy()
        except Exception:
            model_path = None

        if not model_path:
            if not os.path.exists(FALLBACK_MODEL_PATH):
                raise RuntimeError("No model available")
            model_path = FALLBACK_MODEL_PATH

        model = CatBoostRegressor()
        model.load_model(model_path)
        self._model = model
        return model

class WeatherHistoryClient:
    API_URL = "https://archive-api.open-meteo.com/v1/archive"

    @staticmethod
    def load_last_days(lat: float, lon: float, days: int = 60) -> pd.DataFrame:
        end = datetime.now().date()
        start = end - timedelta(days=days)

        response = requests.get(
            WeatherHistoryClient.API_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "daily": ["temperature_2m_mean"],
                "timezone": "auto"
            }
        )
        response.raise_for_status()

        payload = response.json()["daily"]
        df = pd.DataFrame(payload)
        df["time"] = pd.to_datetime(df["time"])
        df["temperature_2m_mean"] = df["temperature_2m_mean"].astype(float)

        return df.sort_values("time").reset_index(drop=True)

class FeatureBuilder:
    TARGET = "temperature_2m_mean"

    @staticmethod
    def build(history: pd.DataFrame, forecast_date: pd.Timestamp) -> pd.DataFrame:
        placeholder = pd.DataFrame({
            "time": [forecast_date],
            FeatureBuilder.TARGET: [np.nan]
        })

        df = (
            pd.concat([history, placeholder])
            .sort_values("time")
            .reset_index(drop=True)
        )

        df["doy"] = df["time"].dt.dayofyear  # вместо df["day_of_year"]
        df["weekday"] = df["time"].dt.weekday
        df["month"] = df["time"].dt.month

        df["season_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
        df["season_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)

        for lag in (1, 2, 3, 7, 14):
            df[f"lag_{lag}"] = df[FeatureBuilder.TARGET].shift(lag)

        for win in (7, 14, 30):
            shifted = df[FeatureBuilder.TARGET].shift(1)
            df[f"mean_{win}"] = shifted.rolling(win).mean()
            df[f"std_{win}"] = shifted.rolling(win).std()

        features = df.iloc[[-1]].fillna(0)

        return features.drop(columns=["time", FeatureBuilder.TARGET], errors="ignore")

app = FastAPI(title=APP_TITLE)

model_loader = ModelLoader()


@app.on_event("startup")
def startup():
    model_loader.load()

@app.post("/predict", response_model=ForecastResponse)
def predict_weather(request: ForecastRequest):
    try:
        model = model_loader.load()
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Model unavailable")

    city = request.city
    geo = CITY_GEO.get(city, CITY_GEO[DEFAULT_CITY])

    history = WeatherHistoryClient.load_last_days(
        lat=geo["lat"],
        lon=geo["lon"]
    )

    forecasts: List[DailyForecast] = []
    rolling_history = history.copy()

    for date_str in request.dates:
        date = pd.to_datetime(date_str)
        X = FeatureBuilder.build(rolling_history, date)
        print("MODEL FEATURES:", model.feature_names_)
        print("INPUT FEATURES:", list(X.columns))
        print(X.dtypes)

        prediction = model.predict(X)

        if isinstance(prediction[0], (list, np.ndarray)):
            value, variance = prediction[0]
            std = float(np.sqrt(variance))
        else:
            value = float(prediction[0])
            std = 0.0

        rolling_history = pd.concat([
            rolling_history,
            pd.DataFrame({
                "time": [date],
                "temperature_2m_mean": [value]
            })
        ])

        forecasts.append(DailyForecast(
            date=date_str,
            temperature=value,
            uncertainty=std,
            lower_bound=value - 1.96 * std,
            upper_bound=value + 1.96 * std
        ))

    return ForecastResponse(city=city, predictions=forecasts)


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}
