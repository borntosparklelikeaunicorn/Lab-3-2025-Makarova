from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class GeoConfig:
    city: str
    lat: float
    lon: float

@dataclass(frozen=True)
class DateRange:
    start: str
    end: str

@dataclass(frozen=True)
class WeatherConfig:
    metrics: List[str]

@dataclass(frozen=True)
class StorageConfig:
    csv_path: str

@dataclass(frozen=True)
class ClearMLConfig:
    project: str
    dataset: str

LOCATION = GeoConfig(
    city="Samara",
    lat=53.2001,
    lon=50.15
)

PERIOD = DateRange(
    start="2020-01-01",
    end="2024-01-01"
)

WEATHER = WeatherConfig(
    metrics=[
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum"
    ]
)

STORAGE = StorageConfig(
    csv_path="weather_history.csv"
)

CLEARML = ClearMLConfig(
    project="Weather_Forecast_Project",
    dataset="Samara_Weather_History"
)
