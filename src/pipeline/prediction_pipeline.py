import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        try:
            print("Before Loading")
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)
            print("After Loading")

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, date, open, high, low, volume, name):
        self.date = date
        self.open = open
        self.high = high
        self.low = low
        self.volume = volume
        self.name = name

    def get_data_as_dataframe(self):
        try:
            df = pd.DataFrame(
                [
                    {
                        "date": pd.to_datetime(self.date),
                        "open": self.open,
                        "high": self.high,
                        "low": self.low,
                        "volume": self.volume,
                        "Name": self.name,
                    }
                ]
            )

            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["day"] = df["date"].dt.day
            df["day_of_week"] = df["date"].dt.dayofweek
            df["week_of_year"] = df["date"].dt.isocalendar().week
            df["quarter"] = df["date"].dt.quarter
            df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
            df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
            df["is_quarter_start"] = df["date"].dt.is_quarter_start.astype(int)
            df["is_quarter_end"] = df["date"].dt.is_quarter_end.astype(int)

            df.drop(columns=["date"], inplace=True)

            data = pd.read_csv("artifacts/data/data.csv")

            data = pd.concat([data, df], ignore_index=True)

            cols = ["open", "high", "low", "close"]
            windows = [5, 10, 20]

            for col in cols:
                for w in windows:
                    ema_col = f"{col}_EMA_{w}"
                    data[ema_col] = data[col].ewm(span=w, adjust=False).mean()

            last_row = data.iloc[[-1]].copy()

            feature_columns = [
                "open",
                "high",
                "low",
                "volume",
                "year",
                "month",
                "day",
                "day_of_week",
                "week_of_year",
                "quarter",
                "is_month_start",
                "is_month_end",
                "is_quarter_start",
                "is_quarter_end",
                "open_EMA_5",
                "open_EMA_10",
                "open_EMA_20",
                "high_EMA_5",
                "high_EMA_10",
                "high_EMA_20",
                "low_EMA_5",
                "low_EMA_10",
                "low_EMA_20",
                "close_EMA_5",
                "close_EMA_10",
                "close_EMA_20",
                "Name",
            ]

            last_row = last_row[feature_columns]

            return last_row

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data = CustomData(
        date="2023-10-01",
        open=150.0,
        high=155.0,
        low=149.0,
        volume=1000000,
        name="AAPL",
    )
    df = data.get_data_as_dataframe()
    print(df)

    pipeline = PredictPipeline()
    prediction = pipeline.predict(df)
    print("Prediction:", prediction)
