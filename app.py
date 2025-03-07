import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="GOOG Stock Predictor", layout="wide")

# Custom CSS for Styling
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
        }
        .stApp {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Load Model
model = load_model('models/stock_predction_model.keras')

# Set Fixed Stock
stock = "GOOG"
start_date = "2014-01-01"
end_date = "2025-03-01"

# Fetch Data
st.markdown("<h1 style='text-align: center; color: blue;'>📈 Google (GOOG) Stock Market Predictor</h1>", unsafe_allow_html=True)
try:
    df = yf.download(stock, start=start_date, end=end_date)

    if df.empty:
        st.error("⚠️ No data found for GOOG. Please check the stock symbol.")
    else:
        st.subheader('Stock Data')
        st.dataframe(df.tail(10))

        # Data Preprocessing
        df_train = df['Close'].iloc[:int(len(df)*0.80)]
        df_test = df['Close'].iloc[int(len(df)*0.80):]

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0,1))

        past_100_days = df_train.tail(100)
        df_test = pd.concat([past_100_days, df_test], ignore_index=True)
        df_test_scaled = scaler.fit_transform(df_test.values.reshape(-1,1))

        # Moving Averages Visualization
        st.subheader('📊 Moving Averages')
        ma_50 = df['Close'].rolling(50).mean()
        ma_100 = df['Close'].rolling(100).mean()
        ma_200 = df['Close'].rolling(200).mean()

        fig, ax = plt.subplots(figsize=(12,6))
        plt.plot(df['Close'], label='Closing Price', color='black')
        plt.plot(ma_50, label='50-Day MA', color='red', linestyle='dashed')
        plt.plot(ma_100, label='100-Day MA', color='blue', linestyle='dashed')
        plt.plot(ma_200, label='200-Day MA', color='green', linestyle='dashed')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)

        # Preparing Data for Prediction
        x, y = [], []
        for i in range(100, df_test_scaled.shape[0]):
            x.append(df_test_scaled[i-100:i])
            y.append(df_test_scaled[i, 0])

        x, y = np.array(x), np.array(y)

        # Predictions
        predictions = model.predict(x)
        predictions = predictions * (1/scaler.scale_)
        y = y * (1/scaler.scale_)

        # Prediction Visualization
        st.subheader('📉 Actual vs Predicted GOOG Stock Prices')
        fig2, ax2 = plt.subplots(figsize=(12,6))
        plt.plot(y, label="Actual Price", color='blue')
        plt.plot(predictions, label="Predicted Price", color='red', linestyle='dashed')
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.grid(True)
        st.pyplot(fig2)

        st.success("✅ Prediction complete!")

except Exception as e:
    st.error(f"⚠️ Error fetching GOOG stock data: {e}")
