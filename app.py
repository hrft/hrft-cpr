import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests

st.set_page_config(layout="wide")
st.title("📈 داشبورد پیش‌بینی قیمت ارز دیجیتال با LSTM")

# انتخاب ارز دیجیتال و بازه زمانی
symbol = st.selectbox("ارز دیجیتال:", ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"])
interval = st.selectbox("بازه زمانی:", ["1d", "1h", "30m"])
lookback_days = st.slider("تعداد روزهای گذشته برای آموزش:", min_value=60, max_value=365, value=180)

# دریافت قیمت زنده از CoinGecko
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


# دریافت داده تاریخی
end_date = datetime.today()
start_date = end_date - timedelta(days=lookback_days)
data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
if data.empty:
    st.warning("❌ داده‌ای برای این تنظیمات پیدا نشد.")
    st.stop()

# آماده‌سازی داده‌ها
data = data[["Close"]]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data)

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# آموزش سریع مدل (یا بارگذاری)
@st.cache_resource
def load_or_train_model():
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    return model

model = load_or_train_model()

# پیش‌بینی ۷ روز آینده
last_60 = scaled[-60:].reshape(1, 60, 1)
future_predictions = []
input_seq = last_60.copy()

for _ in range(7):
    next_pred = model.predict(input_seq)[0][0]
    future_predictions.append(next_pred)
    input_seq = np.append(input_seq[:, 1:, :], [[[next_pred]]], axis=1)

predicted = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# نمایش نمودار
st.subheader("📊 نمودار پیش‌بینی ۷ روز آینده")
fig, ax = plt.subplots()
ax.plot(data.index[-100:], data["Close"].values[-100:], label="قیمت واقعی")
future_dates = [data.index[-1] + timedelta(days=i+1) for i in range(7)]
ax.plot(future_dates, predicted, label="پیش‌بینی", color="red")
ax.legend()
st.pyplot(fig)

