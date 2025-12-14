import requests
import pandas as pd
from clearml import Dataset, Task
from app.settings import LOCATION, PERIOD, WEATHER, STORAGE, CLEARML

ARCHIVE_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"

def request_weather_data() -> pd.DataFrame:
    response = requests.get(
        ARCHIVE_ENDPOINT,
        params={
            "latitude": LOCATION.lat,
            "longitude": LOCATION.lon,
            "start_date": PERIOD.start,
            "end_date": PERIOD.end,
            "daily": WEATHER.metrics,
            "timezone": "auto"
        },
        timeout=30
    )
    response.raise_for_status()
    payload = response.json()["daily"]
    df = pd.DataFrame(payload)
    df["time"] = pd.to_datetime(df["time"])
    df["city"] = LOCATION.city
    return df

def save_csv(df: pd.DataFrame) -> None:
    df.to_csv(STORAGE.csv_path, index=False)

def upload_to_clearml() -> str:
    task = Task.init(project_name=CLEARML.project,
                     task_name="Weather Dataset Registration",
                     task_type=Task.TaskTypes.data_processing)
    dataset = Dataset.create(
        dataset_project=CLEARML.project,
        dataset_name=CLEARML.dataset
    )
    dataset.add_files(STORAGE.csv_path)
    dataset.upload()
    dataset.finalize()
    dataset_id = dataset.id
    task.close()
    return dataset_id

def run_pipeline():
    df = request_weather_data()
    print(f"Weather records downloaded: {df.shape[0]}")
    save_csv(df)
    dataset_id = upload_to_clearml()
    print(f"Dataset uploaded to ClearML with ID: {dataset_id}")

if __name__ == "__main__":
    run_pipeline()
