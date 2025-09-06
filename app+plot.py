# app.py - Advanced seq2seq LSTM multivariate forecasting + candlestick plotting
import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
import requests
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# TensorFlow / Keras
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

st.set_page_config(page_title="Crypto Prediction Dashboard (Seq2Seq)", layout="wide")
st.title("📈 داشبورد پیش‌بینی ارز دیجیتال — کندل + Seq2Seq LSTM")

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.header("⚙️ تنظیمات")
symbol = st.sidebar.selectbox("انتخاب ارز", ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"])
interval = st.sidebar.selectbox("بازه زمانی", ["1d", "1h", "30m"])
period = st.sidebar.selectbox("طول داده تاریخی (yfinance)", ["6mo", "1y", "2y"], index=0)

time_steps = st.sidebar.number_input("تعداد روز/نقاط ورودی (time_steps)", min_value=20, max_value=500, value=60, step=10)
output_steps = st.sidebar.number_input("تعداد قدم‌های خروجی (days to predict)", min_value=1, max_value=30, value=7)
train_toggle = st.sidebar.checkbox("Train model (خاموش = فقط بارگذاری best_model.h5 در صورت وجود)", value=True)

# mapping for future date freq
freq_map = {"1d": "D", "1h": "H", "30m": "30T"}

# -----------------------
# fetch data (yfinance)
# -----------------------
@st.cache_data(ttl=60*30)
def load_data_yf(symbol, period, interval):
    # specify auto_adjust to avoid FutureWarning and progress=False to reduce logs
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    df.dropna(inplace=True)
    return df

with st.spinner("⏳ در حال دریافت داده..."):
    df = load_data_yf(symbol, period, interval)

if df.empty or len(df) < (time_steps + output_steps + 10):
    st.error("داده کافی در دسترس نیست. طول دوره تاریخی را افزایش دهید یا time_steps را کاهش دهید.")
    st.stop()

# -----------------------
# Feature engineering
# -----------------------
data = df.copy()
# Basic indicators
data['return_1'] = data['Close'].pct_change().fillna(0)
data['sma20'] = data['Close'].rolling(window=20, min_periods=1).mean()
data['ema12'] = data['Close'].ewm(span=12, adjust=False).mean()
data['ema26'] = data['Close'].ewm(span=26, adjust=False).mean()
data['macd'] = data['ema12'] - data['ema26']
# RSI (14)
delta = data['Close'].diff().fillna(0)
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
roll_up = up.rolling(14, min_periods=1).mean()
roll_down = down.rolling(14, min_periods=1).mean()
rs = roll_up / (roll_down + 1e-8)
data['rsi'] = 100.0 - (100.0 / (1.0 + rs))
data['volatility'] = data['Close'].rolling(window=20, min_periods=1).std().fillna(0)

# target (we will predict OHLCV)
target_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
# features: include OHLCV + indicators
feature_cols = target_cols + ['sma20', 'macd', 'rsi', 'volatility', 'return_1']

data = data[feature_cols].copy().dropna()
if len(data) < (time_steps + output_steps + 5):
    st.error("بعد از ساخت شاخص‌ها تعداد نمونه کافی نیست؛ دوره تاریخی را افزایش دهید.")
    st.stop()

# -----------------------
# Scaling - fit on training portion only to avoid leakage
# -----------------------
n_total = len(data)
n_train_df = int(n_total * 0.8)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# fit scalers on training portion (by index)
scaler_X.fit(data[feature_cols].iloc[:n_train_df])
scaler_y.fit(data[target_cols].iloc[:n_train_df])

X_all = scaler_X.transform(data[feature_cols])
y_all = scaler_y.transform(data[target_cols])

# -----------------------
# create sequences (multivariate seq2seq)
# -----------------------
X_seqs = []
y_seqs = []
for i in range(time_steps, len(data) - output_steps + 1):
    X_seqs.append(X_all[i - time_steps:i, :])            # shape (time_steps, n_features)
    y_seqs.append(y_all[i:i + output_steps, :])          # shape (output_steps, n_targets)

X = np.array(X_seqs)   # (samples, time_steps, n_features)
y = np.array(y_seqs)   # (samples, output_steps, n_targets)

# train/test split (on samples)
n_samples = X.shape[0]
n_train = int(n_samples * 0.8)

X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

st.sidebar.write(f"نمونه‌ها: کل={n_samples}, آموزش={n_train}, تست={n_samples-n_train}")

# -----------------------
# Build seq2seq model (functional API, avoids input_shape warning)
# -----------------------
n_features = X.shape[2]
n_targets = y.shape[2]

def build_model(time_steps, n_features, output_steps, n_targets, units=128):
    encoder_inputs = Input(shape=(time_steps, n_features))
    encoder_lstm = LSTM(units, return_state=True)
    _, state_h, state_c = encoder_lstm(encoder_inputs)
    # use state_h repeated
    decoder_inputs = RepeatVector(output_steps)(state_h)
    decoder_lstm = LSTM(units, return_sequences=True)(decoder_inputs, initial_state=[state_h, state_c])
    decoder_outputs = TimeDistributed(Dense(n_targets))(decoder_lstm)
    model = Model(encoder_inputs, decoder_outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model_path = "best_model.h5"
model = None

if (not train_toggle) and os.path.exists(model_path):
    try:
        model = load_model(model_path)
        st.success("مدل از فایل بارگذاری شد: best_model.h5")
    except Exception as e:
        st.warning("بارگذاری مدل موفق نبود، مدل را آموزش می‌دهیم. خطا: " + str(e))
        model = None

if model is None:
    model = build_model(time_steps, n_features, output_steps, n_targets, units=128)

if train_toggle:
    # callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]
    epochs = st.sidebar.number_input("epochs", min_value=1, max_value=200, value=50)
    batch_size = st.sidebar.number_input("batch_size", min_value=8, max_value=256, value=32)

    with st.spinner("⏳ در حال آموزش مدل (ممکن است طولانی باشد)..."):
        history = model.fit(X_train, y_train,
                            validation_split=0.1,
                            epochs=int(epochs),
                            batch_size=int(batch_size),
                            callbacks=callbacks,
                            verbose=1)
    st.success("✅ آموزش مدل کامل شد. بهترین مدل در best_model.h5 ذخیره شد.")
else:
    # ensure model is compiled/available
    if not os.path.exists(model_path):
        st.warning("فایل best_model.h5 پیدا نشد. مدل آموزش داده خواهد شد.")
        # train minimal quick if file not found
        with st.spinner("آموزش سریع مدل..."):
            model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
            model.save(model_path)
            st.success("مدل سریع آموزش داده و ذخیره شد.")
    else:
        model = load_model(model_path)
        st.success("مدل بارگذاری شد.")

# -----------------------
# Evaluate on test (optional)
# -----------------------
with st.spinner("در حال ارزیابی مدل روی دادهٔ تست..."):
    y_pred_test_scaled = model.predict(X_test)  # shape (samples, output_steps, n_targets)
    # inverse transform for y_test and y_pred
    y_test_2d = y_test.reshape(-1, n_targets)
    y_pred_2d = y_pred_test_scaled.reshape(-1, n_targets)
    y_test_inv = scaler_y.inverse_transform(y_test_2d).reshape(y_test.shape)
    y_pred_inv = scaler_y.inverse_transform(y_pred_2d).reshape(y_pred_test_scaled.shape)

    # compute MAE/RMSE for Close (target index)
    close_idx = target_cols.index('Close')
    mae_close = mean_absolute_error(y_test_inv[:,:,close_idx].ravel(), y_pred_inv[:,:,close_idx].ravel())
    rmse_close = np.sqrt(mean_squared_error(y_test_inv[:,:,close_idx].ravel(), y_pred_inv[:,:,close_idx].ravel()))

st.sidebar.metric("Test MAE (Close)", f"{mae_close:.4f}")
st.sidebar.metric("Test RMSE (Close)", f"{rmse_close:.4f}")

# -----------------------
# Predict next output_steps using last time_steps window
# -----------------------
last_X = X_all[-time_steps:, :]  # scaled features
input_for_pred = last_X.reshape(1, time_steps, n_features)
pred_scaled = model.predict(input_for_pred)  # shape (1, output_steps, n_targets)
pred_scaled_2d = pred_scaled.reshape(output_steps, n_targets)
pred_inv = scaler_y.inverse_transform(pred_scaled_2d)  # (output_steps, n_targets)
pred_df = pd.DataFrame(pred_inv, columns=target_cols)

# create future dates based on interval
last_timestamp = data.index[-1]
freq = freq_map.get(interval, "D")
future_dates = pd.date_range(start=last_timestamp + pd.Timedelta(1, unit=freq[0].upper()), periods=output_steps, freq=freq)
pred_df.index = future_dates

# -----------------------
# Plot candlesticks: actual (last N) + predicted (next output_steps)
# -----------------------
st.subheader("📊 نمودار کندلی — واقعی (آبی) و پیش‌بینی (نارنجی)")

plot_n = min(len(data), 200)
actual_df = data[target_cols].iloc[-plot_n:].copy()
# ensure index is datetime
actual_df.index = pd.to_datetime(actual_df.index)

fig = go.Figure()

# actual candles
fig.add_trace(go.Candlestick(
    x=actual_df.index,
    open=actual_df['Open'],
    high=actual_df['High'],
    low=actual_df['Low'],
    close=actual_df['Close'],
    name='Actual',
    increasing=dict(line=dict(color='deepskyblue')),
    decreasing=dict(line=dict(color='royalblue'))
))

# predicted candles (plotted as separate candlestick trace)
fig.add_trace(go.Candlestick(
    x=pred_df.index,
    open=pred_df['Open'],
    high=pred_df['High'],
    low=pred_df['Low'],
    close=pred_df['Close'],
    name='Predicted',
    increasing=dict(line=dict(color='orange')),
    decreasing=dict(line=dict(color='darkorange'))
))

# optionally draw connecting line between last actual close and first predicted open
fig.add_trace(go.Scatter(
    x=[actual_df.index[-1], pred_df.index[0]],
    y=[actual_df['Close'].values[-1], pred_df['Open'].values[0]],
    mode='lines',
    name='Bridge',
    line=dict(color='white', width=1, dash='dash')
))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    height=650
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Show predicted table and download
# -----------------------
st.subheader("🔮 پیش‌بینی ۷ قدم آینده (OHLCV)")
st.dataframe(pred_df)

csv = pred_df.to_csv().encode('utf-8')
st.download_button("⬇️ دانلود CSV پیش‌بینی", csv, "predicted_ohlcv.csv", "text/csv")

st.info("نکته: برای دستیابی به دقت بهتر، موارد زیر را امتحان کنید:\n"
        "- افزایش تعداد epoch ها و/یا اندازه شبکه (units)\n"
        "- استفاده از GPU (برای سرعت آموزش و امکان افزایش epochs)\n"
        "- تنظیم هایپرپارامترها (batch_size, learning rate و...)\n"
        "- اضافه کردن ویژگی‌های بیشتر (مثلاً داده‌های بازار/اندیکاتورهای دیگر)\n"
        "- اعتبارسنجی پیشرفته‌تر و walk-forward validation")
