import streamlit as st
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

st.title("S&P 500 Stock Price Prediction")
st.write("Enter the details of the stock to predict its closing price.")

date = st.date_input("Date")
open_price = st.number_input("Open Price", min_value=0.0)
high_price = st.number_input("High Price", min_value=0.0)
low_price = st.number_input("Low Price", min_value=0.0)
volume = st.number_input("Volume", min_value=0)
name = st.text_input("Stock Name")
if st.button("Predict"):
    data = CustomData(
        date=date,
        open=open_price,
        high=high_price,
        low=low_price,
        volume=volume,
        name=name
    )
    pred_df = data.get_data_as_dataframe()
    st.write("Input Data:")
    st.dataframe(pred_df)

    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    st.success(f"The predicted closing price is: {results[0]}")