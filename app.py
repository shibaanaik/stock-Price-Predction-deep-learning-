import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model('stock_predction_model.keras')


st.header('Stock Market Predictor')

stock =st.text_input('Enter Stock Symnbol', 'GOOG')
start = '2014-01-01'
end = '2025-03-01'

df = yf.download(stock, start ,end)

st.subheader('Stock Data')
st.write(df)

df_train = pd.DataFrame(df.Close[0: int(len(df)*0.80)])
df_test = pd.DataFrame(df.Close[int(len(df)*0.80): len(df)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = df_train.tail(100)
df_test = pd.concat([pas_100_days, df_test], ignore_index=True)
df_test_scale = scaler.fit_transform(df_test)

st.subheader('Price vs MA50')
ma_50_days = df.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(df.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = df.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(df.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = df.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(df.Close, 'g')
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, df_test_scale.shape[0]):
    x.append(df_test_scale[i-100:i])
    y.append(df_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

s = 1/scaler.scale_

predict = predict * s
y = y * s

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)
