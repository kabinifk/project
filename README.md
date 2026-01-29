```import numpy as np```
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

time_steps = 1000
t = np.arange(time_steps)

trend = 0.05 * t
seasonal_1 = 10 * np.sin(2 * np.pi * t / 50)
seasonal_2 = 5 * np.sin(2 * np.pi * t / 200)
noise = np.random.normal(0, 2, time_steps)

target = trend + seasonal_1 + seasonal_2 + noise
exog_1 = np.cos(2 * np.pi * t / 50)
exog_2 = np.random.normal(0, 1, time_steps)

df = pd.DataFrame({
    "target": target,
    "exog_1": exog_1,
    "exog_2": exog_2
})

df.head()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

def create_sequences(data, window=30):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window, 0])
    return np.array(X), np.array(y)

WINDOW = 30
X, y = create_sequences(scaled, WINDOW)

train_size = int(0.7 * len(X))
val_size = int(0.85 * len(X))

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:val_size], y[train_size:val_size]
X_test, y_test = X[val_size:], y[val_size:]
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

baseline = Sequential([
    LSTM(64, input_shape=(WINDOW, X.shape[2])),
    Dense(1)
])

baseline.compile(optimizer="adam", loss="mse")
baseline.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
from tensorflow.keras.layers import Input, Attention
from tensorflow.keras.models import Model



inputs = Input(shape=(WINDOW , X.shape[2]))
lstm_out = LSTM(64, return_sequences=True)(inputs)

attention = Attention()([lstm_out, lstm_out])

context = GlobalAveragePooling1D()(attention)


output = Dense(1)(context)

attn_model = Model(inputs, output)
attn_model.compile(optimizer="adam", loss="mse")
attn_model.fit(X_train, y_train, epochs=15, validation_data=(X_val, y_val))
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

baseline_pred = baseline.predict(X_test)
attn_pred = attn_model.predict(X_test)

print("Baseline RMSE:", rmse(y_test, baseline_pred))
print("Attention RMSE:", rmse(y_test, attn_pred))

print("Baseline MAPE:", mape(y_test, baseline_pred))
print("Attention MAPE:", mape(y_test, attn_pred))
sample_input = X_test[0:1]
attention_weights = attn_model.layers[2]([sample_input, sample_input])

print("Attention shape:", attention_weights.shape)
