import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

# Load Model
model = load_model('models/stock_predction_model.keras')

# Apply CSS Styles
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
        }
        h1, h2, h3, h4, h5, h6 {
            color: black !important;
            text-align: center;
            font-weight: bold;
        }
        .stApp {
            background-color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# Styled Title
st.markdown("<h1>📈 Google (GOOG) Stock Market Predictor</h1>", unsafe_allow_html=True)

# Download Stock Data
stock = "GOOG"
start = '2014-01-01'
end = '2025-03-01'
df = yf.download(stock, start, end)

# Stock Data Section
st.markdown("### 📊 Stock Data")
st.write(df)

# Splitting Data
df_train = pd.DataFrame(df.Close[0: int(len(df)*0.80)])
df_test = pd.DataFrame(df.Close[int(len(df)*0.80): len(df)])

# Scaling Data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
pas_100_days = df_train.tail(100)
df_test = pd.concat([pas_100_days, df_test], ignore_index=True)
df_test_scale = scaler.fit_transform(df_test)

# Price vs MA50
st.markdown("### 📈 Price vs MA50")
ma_50_days = df.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label="MA 50 Days")
plt.plot(df.Close, 'g', label="Stock Price")
plt.legend()
plt.show()
st.pyplot(fig1)

# Price vs MA50 vs MA100
st.markdown("### 📈 Price vs MA50 vs MA100")
ma_100_days = df.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label="MA 50 Days")
plt.plot(ma_100_days, 'b', label="MA 100 Days")
plt.plot(df.Close, 'g', label="Stock Price")
plt.legend()
plt.show()
st.pyplot(fig2)

# Price vs MA100 vs MA200
st.markdown("### 📈 Price vs MA100 vs MA200")
ma_200_days = df.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r', label="MA 100 Days")
plt.plot(ma_200_days, 'b', label="MA 200 Days")
plt.plot(df.Close, 'g', label="Stock Price")
plt.legend()
plt.show()
st.pyplot(fig3)

# Preparing Data for Prediction
x = []
y = []
for i in range(100, df_test_scale.shape[0]):
    x.append(df_test_scale[i-100:i])
    y.append(df_test_scale[i,0])

x,y = np.array(x), np.array(y)

# Model Prediction
predict = model.predict(x)
s = 1/scaler.scale_
predict = predict * s
y = y * s

# Prediction Graph
st.markdown("### 🎯 Original Price vs Predicted Price")
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Actual Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)
