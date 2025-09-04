import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
import requests
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Crypto Prediction Dashboard", layout="wide")

st.title("📈 داشبورد پیش‌بینی قیمت ارزهای دیجیتال")

# ===============================
# تنظیمات سایدبار
# ===============================
st.sidebar.header("⚙️ تنظیمات")

symbol = st.sidebar.selectbox("انتخاب ارز", ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"])
interval = st.sidebar.selectbox("بازه زمانی", ["1d", "1h", "30m"])
period = st.sidebar.selectbox("طول داده تاریخی", ["6mo", "1y", "2y"], index=0)

# ===============================
# دریافت داده تاریخی از Yahoo Finance
# ===============================
@st.cache_data
def load_data(symbol, period, interval):
    data = yf.download(symbol, period=period, interval=interval)
    return data

data = load_data(symbol, period, interval)

# ===============================
# مدل LSTM برای پیش‌بینی
# ===============================
def train_and_predict(data, days=7):
    df = data[["Close"]].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # آماده‌سازی داده
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # مدل LSTM
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=3, batch_size=32, verbose=0)

    # پیش‌بینی روزهای آینده
    last_60 = scaled_data[-60:]
    input_data = last_60.reshape(1, 60, 1)

    predictions = []
    for _ in range(days):
        pred = model.predict(input_data, verbose=0)
        predictions.append(pred[0, 0])
        input_data = np.append(input_data[:, 1:, :], [[pred]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

predicted = train_and_predict(data, days=7)

# ===============================
# دریافت قیمت زنده از CoinGecko
# ===============================
st.sidebar.title("💰 قیمت زنده (CoinGecko)")

def fetch_coingecko_price(symbol="bitcoin", vs="usd"):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies={vs}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        return data[symbol][vs]
    except:
        return None

symbol_map = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "BNB-USD": "binancecoin",
    "SOL-USD": "solana"
}

live_price = fetch_coingecko_price(symbol_map[symbol], "usd")
if live_price:
    st.sidebar.metric(label=f"💲 قیمت لحظه‌ای {symbol}", value=f"{live_price:,.2f} USD")
else:
    st.sidebar.warning("❌ دریافت قیمت زنده ممکن نیست.")

# ===============================
# نمودار کندلی + پیش‌بینی
# ===============================
st.subheader("📊 نمودار کندلی + پیش‌بینی ۷ روز آینده")

fig = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="قیمت واقعی"
)])

# اضافه کردن پیش‌بینی
future_dates = [data.index[-1] + timedelta(days=i+1) for i in range(7)]
fig.add_trace(go.Scatter(
    x=future_dates,
    y=predicted.flatten(),
    mode="lines+markers",
    name="پیش‌بینی",
    line=dict(color="red", width=2)
))

fig.update_layout(
    xaxis_title="تاریخ",
    yaxis_title="قیمت (USD)",
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    height=600
)

st.plotly_chart(fig, use_container_width=True)
