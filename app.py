# app.py
# Final pipeline: BiLSTM + Attention seq2seq, show last 2 actual rows + 5-day forecast (OHLCV only, table)
import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.layers import (Input, LSTM, Dense, RepeatVector, TimeDistributed,
                                     Bidirectional, Concatenate, Dot, Activation)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="Final Crypto Forecast (Table)", layout="wide")
st.title("🔮 Final Forecast — BiLSTM + Attention (Table only)")

# ---------------------------
# Sidebar: user controls
# ---------------------------
st.sidebar.header("⚙️ تنظیمات")

symbol = st.sidebar.selectbox("انتخاب ارز", ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"])
interval = st.sidebar.selectbox("بازه زمانی", ["1d", "1h"], index=0)
period = st.sidebar.selectbox("طول داده تاریخی (yfinance)", ["6mo", "1y", "2y"], index=1)

time_steps = st.sidebar.number_input("طول پنجره ورودی (time_steps)", min_value=30, max_value=300, value=60, step=10)
output_steps = st.sidebar.number_input("تعداد قدم‌های خروجی (پیش‌بینی)", min_value=1, max_value=30, value=5)
train_toggle = st.sidebar.checkbox("Train model now", value=True)

# model hyperparams (defaults balanced for CPU)
units = st.sidebar.number_input("units (شبکه BiLSTM)", min_value=32, max_value=512, value=128, step=32)
epochs = st.sidebar.number_input("epochs (توصیه GPU برای خیلی بالا)", min_value=5, max_value=500, value=60)
batch_size = st.sidebar.number_input("batch_size", min_value=8, max_value=512, value=32)

model_path = "best_model_final.h5"

# ---------------------------
# Helpers: load data
# ---------------------------
@st.cache_data(ttl=60*30)
def load_data(symbol, period, interval):
    # set auto_adjust explicitly to avoid future warnings
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    df.dropna(inplace=True)
    return df

with st.spinner("⏳ دریافت داده..."):
    df_raw = load_data(symbol, period, interval)

if df_raw.empty:
    st.error("داده‌ای دریافت نشد — پارامترها را تغییر دهید.")
    st.stop()

# ---------------------------
# Feature engineering
# ---------------------------
# target columns (we will predict OHLCV)
target_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

df = df_raw.copy()
# basic technical features to enrich inputs
df['return_1'] = df['Close'].pct_change().fillna(0)
df['sma20'] = df['Close'].rolling(window=20, min_periods=1).mean()
df['ema12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['ema26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['macd'] = df['ema12'] - df['ema26']
delta = df['Close'].diff().fillna(0)
up = delta.clip(lower=0); down = -1 * delta.clip(upper=0)
roll_up = up.rolling(14, min_periods=1).mean()
roll_down = down.rolling(14, min_periods=1).mean()
rs = roll_up / (roll_down + 1e-8)
df['rsi'] = 100.0 - (100.0 / (1.0 + rs))
df['volatility'] = df['Close'].rolling(window=20, min_periods=1).std().fillna(0)

# feature columns = OHLCV + indicators
feature_cols = target_cols + ['sma20', 'macd', 'rsi', 'volatility', 'return_1']

data = df[feature_cols].dropna()
if len(data) < (time_steps + output_steps + 5):
    st.error("دادهٔ کافی بعد از محاسبهٔ ویژگی‌ها در دسترس نیست — دوره را افزایش دهید یا time_steps را کاهش دهید.")
    st.stop()

# ---------------------------
# Scaling (fit only on train portion to avoid leakage)
# ---------------------------
n_total = len(data)
n_train_df = int(n_total * 0.8)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# fit scalers on training portion (features & targets)
scaler_X.fit(data[feature_cols].iloc[:n_train_df])
scaler_y.fit(data[target_cols].iloc[:n_train_df])

X_all = scaler_X.transform(data[feature_cols])
y_all = scaler_y.transform(data[target_cols])

# ---------------------------
# create seq2seq dataset (multivariate)
# ---------------------------
X_seqs = []
y_seqs = []
for i in range(time_steps, len(data) - output_steps + 1):
    X_seqs.append(X_all[i - time_steps:i, :])
    y_seqs.append(y_all[i:i + output_steps, :])

X = np.array(X_seqs)   # (samples, time_steps, n_features)
y = np.array(y_seqs)   # (samples, output_steps, n_targets)

n_samples = X.shape[0]
n_train = int(n_samples * 0.8)

X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

n_features = X.shape[2]
n_targets = y.shape[2]

st.sidebar.write(f"نمونه‌ها: کل={n_samples}, آموزش={n_train}, تست={n_samples-n_train}")

# ---------------------------
# Build BiLSTM + Attention model
# ---------------------------
def build_bi_lstm_attention(time_steps, n_features, output_steps, n_targets, units=128):
    encoder_inputs = Input(shape=(time_steps, n_features))                       # (batch, time_steps, n_features)
    encoder_bi = tf.keras.layers.Bidirectional(LSTM(units, return_sequences=True))(encoder_inputs)  # (batch, time_steps, 2*units)
    encoder_last = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(encoder_bi)    # (batch, 2*units)

    decoder_inputs = RepeatVector(output_steps)(encoder_last)                   # (batch, output_steps, 2*units)
    decoder_lstm = LSTM(units*2, return_sequences=True)(decoder_inputs)        # (batch, output_steps, 2*units)

    # Attention
    # score = decoder_lstm dot encoder_bi (on features dim)
    score = Dot(axes=[2, 2])([decoder_lstm, encoder_bi])                        # (batch, output_steps, time_steps)
    attention_weights = Activation('softmax')(score)
    context = Dot(axes=[2,1])([attention_weights, encoder_bi])                  # (batch, output_steps, 2*units)

    decoder_combined = Concatenate(axis=-1)([context, decoder_lstm])           # (batch, output_steps, 4*units)
    outputs = TimeDistributed(Dense(n_targets))(decoder_combined)

    model = Model(encoder_inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# try loading model if exists and user doesn't want training
model = None
if (not train_toggle) and os.path.exists(model_path):
    try:
        model = load_model(model_path)
        st.success("مدل از فایل بارگذاری شد.")
    except Exception as e:
        st.warning("بارگذاری مدل موفق نبود، مدل جدید ساخته خواهد شد: " + str(e))
        model = None

if model is None:
    model = build_bi_lstm_attention(time_steps, n_features, output_steps, n_targets, units=units)

# ---------------------------
# Training with callbacks (EarlyStopping, ReduceLR, ModelCheckpoint)
# ---------------------------
if train_toggle:
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]
    with st.spinner("⏳ در حال آموزش مدل (ممکن است زمان ببرد)..."):
        history = model.fit(X_train, y_train,
                            validation_split=0.1,
                            epochs=int(epochs),
                            batch_size=int(batch_size),
                            callbacks=callbacks,
                            verbose=1)
    st.success("✅ آموزش مدل کامل شد و بهترین وزن در best_model_final.h5 ذخیره شد.")
else:
    # if model file missing, do a quick train to have usable model
    if not os.path.exists(model_path):
        st.warning("مدل ذخیره‌شده پیدا نشد — آموزش سریع انجام می‌شود.")
        model.fit(X_train, y_train, epochs=5, batch_size=int(batch_size), verbose=0)
        model.save(model_path)
    else:
        model = load_model(model_path)
        st.success("مدل بارگذاری شد.")

# ---------------------------
# Evaluate on test set (MAE/RMSE for Close)
# ---------------------------
y_pred_test = model.predict(X_test)
y_test_2d = y_test.reshape(-1, n_targets)
y_pred_2d = y_pred_test.reshape(-1, n_targets)

y_test_inv = scaler_y.inverse_transform(y_test_2d).reshape(y_test.shape)
y_pred_inv = scaler_y.inverse_transform(y_pred_2d).reshape(y_pred_test.shape)

close_idx = target_cols.index('Close')
mae_close = mean_absolute_error(y_test_inv[:,:,close_idx].ravel(), y_pred_inv[:,:,close_idx].ravel())
rmse_close = np.sqrt(mean_squared_error(y_test_inv[:,:,close_idx].ravel(), y_pred_inv[:,:,close_idx].ravel()))

st.sidebar.metric("MAE (Close)", f"{mae_close:.6f}")
st.sidebar.metric("RMSE (Close)", f"{rmse_close:.6f}")

# ---------------------------
# Predict next steps using last window
# ---------------------------
last_X = X_all[-time_steps:, :]
input_for_pred = last_X.reshape(1, time_steps, n_features)
pred_scaled = model.predict(input_for_pred)  # shape (1, output_steps, n_targets)
pred_2d = pred_scaled.reshape(output_steps, n_targets)
pred_inv = scaler_y.inverse_transform(pred_2d)  # (output_steps, n_targets)
pred_df = pd.DataFrame(pred_inv, columns=target_cols)

# build future index: use same frequency as input interval
if interval == "1d":
    freq = "D"
elif interval == "1h":
    freq = "H"
else:
    freq = "D"

last_ts = pd.to_datetime(data.index[-1])
future_index = pd.date_range(start=last_ts + pd.Timedelta(1, unit=freq), periods=output_steps, freq=freq)
pred_df.index = future_index

# ---------------------------
# Prepare final table: last 2 actual rows + predicted rows
# ---------------------------
# get last 2 actual rows (ensure they are from 'data' (features df) because it has consistent columns)
actual_last2 = data[target_cols].iloc[-2:].copy()
# set index names to recent timestamps (they already are)
# concat actual_last2 then predicted
final_table = pd.concat([actual_last2, pred_df])

# format numbers
final_table_rounded = final_table.round(4)

st.subheader(f"🔮 جدول خروجی: ۲ ردیف واقعی اخیر + {output_steps} ردیف پیش‌بینی (OHLCV)")
st.dataframe(final_table_rounded)

# download button
csv = final_table_rounded.to_csv().encode('utf-8')
st.download_button("⬇️ دانلود CSV پیش‌بینی", csv, "final_prediction_table.csv", "text/csv")

# ---------------------------
# Advice & Next steps (for reducing error further)
# ---------------------------
st.markdown("### 🔧 نکات برای کاهش بیشتر خطا (گام‌های بعدی)")
st.markdown("""
- استفاده از **GPU** و افزایش `epochs` و `units`.
- **Walk-forward validation** و بهینه‌سازی هایپرپارامتر (Optuna یا Keras-Tuner).
- افزودن داده‌های بیشتر و ویژگی‌های بیرونی (on-chain metrics, funding rate, orderbook depth).
- استفاده از **ensemble** (BiLSTM+Transformer+XGBoost روی ویژگی‌های استخراج‌شده).
- اجرای **feature selection** و استفاده از regularization و dropout تنظیم‌شده.
""")
