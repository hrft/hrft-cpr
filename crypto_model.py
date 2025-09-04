
---

## 📌 فایل ۴: `crypto_model.py` (اختیاری)
```python
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# دریافت داده‌های 6 ماه گذشته
end_date = datetime.today()
start_date = end_date - timedelta(days=180)
btc_data = yf.download('BTC-USD', start=start_date, end=end_date, interval='1d')

btc_data = btc_data[['Close']]

# نرمال‌سازی
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(btc_data)

def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i+time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# ساخت مدل
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32)

# ذخیره مدل
model.save("btc_lstm_model.h5")

