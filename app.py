import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

import streamlit as st
from src.pipeline.prediction_pipeline import PredictPipeline

st.title("S&P 500 Stock Price Prediction")
st.write("Enter the details of the stock to predict its closing price.")

# User inputs
date = st.date_input("Date")
open_price = st.number_input("Open Price", min_value=0.0)
high_price = st.number_input("High Price", min_value=0.0)
low_price = st.number_input("Low Price", min_value=0.0)
volume = st.number_input("Volume", min_value=0)
name = st.text_input("Stock Name")

if st.button("Predict"):
    # Create initial DataFrame
    df = pd.DataFrame(
        [
            {
                "date": pd.to_datetime(date),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "volume": volume,
                "Name": name,
            }
        ]
    )

    # Date features
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

    # Drop date column
    df.drop(columns=["date"], inplace=True)

    # Compute EMA features (self-contained)
    # Since we don't have history, EMA = current value
    for col in ["open", "high", "low", "close"]:
        for w in [5, 10, 20]:
            if col == "close":
                df[f"{col}_EMA_{w}"] = open_price  # placeholder for first EMA
            else:
                df[f"{col}_EMA_{w}"] = df[col] if col in df else open_price

    # Ensure all columns expected by the preprocessor are present
    expected_cols = [
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
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0  # fill missing columns with 0

    # Reorder columns to match preprocessor
    df = df[expected_cols]

    st.write("Processed Input Data:")
    st.dataframe(df)

    # Predict
    predict_pipeline = PredictPipeline()
    result = predict_pipeline.predict(df)

    st.success(f"The predicted closing price is: {result[0]:.2f}")
