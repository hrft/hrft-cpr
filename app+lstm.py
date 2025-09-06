# app.py - Seq2Seq LSTM multivariate forecasting, no chart, only table
import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

st.set_page_config(page_title="Crypto Forecast (Table Only)", layout="wide")
st.title("🔮 پیش‌بینی ارز دیجیتال (فقط جدول)")

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.header("⚙️ تنظیمات")
symbol = st.sidebar.selectbox("انتخاب ارز", ["BTC-USD", "ETH-USD", "BNB-USD"])
interval = st.sidebar.selectbox("بازه زمانی", ["1d", "1h"], index=0)
period = st.sidebar.selectbox("طول داده تاریخی", ["6mo", "1y", "2y"], index=1)

time_steps = st.sidebar.number_input("طول ورودی (time_steps)", min_value=30, max_value=300, value=60, step=10)
output_steps = st.sidebar.number_input("قدم‌های پیش‌بینی", min_value=1, max_value=30, value=7)
train_toggle = st.sidebar.checkbox("Train model", value=True)

# -----------------------
# fetch data
# -----------------------
@st.cache_data(ttl=60*30)
def load_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    df.dropna(inplace=True)
    return df

with st.spinner("⏳ دریافت داده..."):
    df = load_data(symbol, period, interval)

if df.empty:
    st.error("❌ داده دریافت نشد.")
    st.stop()

target_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
feature_cols = target_cols.copy()

data = df[feature_cols].copy().dropna()
if len(data) < (time_steps + output_steps + 5):
    st.error("داده کافی نیست. بازه تاریخی را بیشتر کنید.")
    st.stop()

# -----------------------
# Scaling
# -----------------------
n_total = len(data)
n_train_df = int(n_total * 0.8)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_X.fit(data[feature_cols].iloc[:n_train_df])
scaler_y.fit(data[target_cols].iloc[:n_train_df])

X_all = scaler_X.transform(data[feature_cols])
y_all = scaler_y.transform(data[target_cols])

# -----------------------
# create sequences
# -----------------------
X_seqs, y_seqs = [], []
for i in range(time_steps, len(data) - output_steps + 1):
    X_seqs.append(X_all[i - time_steps:i, :])
    y_seqs.append(y_all[i:i + output_steps, :])

X = np.array(X_seqs)
y = np.array(y_seqs)

n_samples = X.shape[0]
n_train = int(n_samples * 0.8)

X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

n_features = X.shape[2]
n_targets = y.shape[2]

# -----------------------
# Build seq2seq model
# -----------------------
def build_model(time_steps, n_features, output_steps, n_targets, units=256, dropout=0.2):
    encoder_inputs = Input(shape=(time_steps, n_features))
    encoder_lstm, state_h, state_c = LSTM(units, return_state=True, dropout=dropout)(encoder_inputs)
    decoder_inputs = RepeatVector(output_steps)(state_h)
    decoder_lstm = LSTM(units, return_sequences=True, dropout=dropout)(decoder_inputs, initial_state=[state_h, state_c])
    decoder_outputs = TimeDistributed(Dense(n_targets))(decoder_lstm)
    model = Model(encoder_inputs, decoder_outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model_path = "best_model.h5"
model = None

if (not train_toggle) and os.path.exists(model_path):
    try:
        model = load_model(model_path)
        st.success("مدل از فایل ذخیره‌شده بارگذاری شد.")
    except:
        model = None

if model is None:
    model = build_model(time_steps, n_features, output_steps, n_targets)

if train_toggle:
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]
    epochs = st.sidebar.number_input("epochs", 10, 300, 100)
    batch_size = st.sidebar.number_input("batch_size", 8, 256, 32)

    with st.spinner("⏳ آموزش مدل..."):
        model.fit(X_train, y_train,
                  validation_split=0.1,
                  epochs=int(epochs),
                  batch_size=int(batch_size),
                  callbacks=callbacks,
                  verbose=1)
    st.success("✅ آموزش کامل شد و مدل ذخیره شد.")

# -----------------------
# Evaluate
# -----------------------
y_pred_test_scaled = model.predict(X_test)
y_test_2d = y_test.reshape(-1, n_targets)
y_pred_2d = y_pred_test_scaled.reshape(-1, n_targets)

y_test_inv = scaler_y.inverse_transform(y_test_2d).reshape(y_test.shape)
y_pred_inv = scaler_y.inverse_transform(y_pred_2d).reshape(y_pred_test_scaled.shape)

close_idx = target_cols.index('Close')
mae_close = mean_absolute_error(y_test_inv[:,:,close_idx].ravel(), y_pred_inv[:,:,close_idx].ravel())
rmse_close = np.sqrt(mean_squared_error(y_test_inv[:,:,close_idx].ravel(), y_pred_inv[:,:,close_idx].ravel()))

st.sidebar.metric("MAE (Close)", f"{mae_close:.4f}")
st.sidebar.metric("RMSE (Close)", f"{rmse_close:.4f}")

# -----------------------
# Predict future
# -----------------------
last_X = X_all[-time_steps:, :]
input_for_pred = last_X.reshape(1, time_steps, n_features)
pred_scaled = model.predict(input_for_pred)
pred_inv = scaler_y.inverse_transform(pred_scaled.reshape(output_steps, n_targets))

pred_df = pd.DataFrame(pred_inv, columns=target_cols)
pred_df.index = pd.date_range(start=data.index[-1], periods=output_steps+1, freq="D")[1:]

# -----------------------
# Show table + download
# -----------------------
st.subheader("🔮 پیش‌بینی OHLCV (جدول)")
st.dataframe(pred_df)

csv = pred_df.to_csv().encode('utf-8')
st.download_button("⬇️ دانلود CSV", csv, "predicted.csv", "text/csv")
