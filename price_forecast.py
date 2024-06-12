import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from reservoirpy.nodes import Reservoir, Ridge

data = pd.read_csv("data.csv")
data = data[["Date", "Price"]]
data["Date"] = pd.to_datetime(data["Date"], format="%d-%b-%y")

data.head()

train_data, test_data = train_test_split(data, test_size=0.1, shuffle=False)

scaler = MinMaxScaler()
train_data["Price"] = scaler.fit_transform(
    train_data["Price"].values.reshape(-1, 1)
).ravel()
test_data["Price"] = scaler.transform(test_data["Price"].values.reshape(-1, 1)).ravel()


def create_sequences(data, window_size=500):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size : i])
        y.append(data[i])
    return np.array(X), np.array(y)


X_train, y_train = create_sequences(train_data["Price"].values)
X_test, y_test = create_sequences(test_data["Price"].values)


models = []

# MLP
mlp_model = Sequential()
mlp_model.add(Dense(128, input_dim=X_train.shape[1], activation="relu"))
mlp_model.add(Dense(32, activation="relu"))
mlp_model.add(Dense(1))
mlp_model.compile(optimizer="adam", loss="mse")
mlp_model.fit(X_train, y_train, epochs=50, batch_size=32)
models.append(("MLP", mlp_model))

# GRU
gru_model = Sequential()
gru_model.add(GRU(128, input_shape=(X_train.shape[1], 1)))
gru_model.add(Dense(1))
gru_model.compile(optimizer="adam", loss="mse")
gru_model.fit(X_train, y_train, epochs=50, batch_size=32)
models.append(("GRU", gru_model))

# LSTM
lstm_model = Sequential()
lstm_model.add(LSTM(128, input_shape=(X_train.shape[1], 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer="adam", loss="mse")
lstm_model.fit(X_train, y_train, epochs=50, batch_size=32)
models.append(("LSTM", lstm_model))

# RNN
rnn_model = Sequential()
rnn_model.add(SimpleRNN(128, input_shape=(X_train.shape[1], 1)))
rnn_model.add(Dense(1))
rnn_model.compile(optimizer="adam", loss="mse")
rnn_model.fit(X_train, y_train, epochs=50, batch_size=32)
models.append(("RNN", rnn_model))

# XGBoost
xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)
models.append(("XGBoost", xgb_model))

# LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
models.append(("LinearRegression", linear_model))

# SVR
svr_model = SVR(kernel="rbf", gamma="scale", C=1.0, epsilon=0.1)
svr_model.fit(X_train, y_train)
models.append(("SVR", svr_model))

# ESN
# reservoir = Reservoir(reservoir_size=1000, lr=0.3, sr=0.8)
# ridge = Ridge(ridge=1e-7)
# esn = reservoir >> ridge
# esn.fit(X_train, y_train.reshape(-1, 1))
# models.append(("ESN", esn))

results = []
for name, model in models:
    if name in ["MLP", "XGBoost", "LinearRegression", "SVR"]:
        y_pred = model.predict(X_test)
    elif name == "ESN":
        y_pred = model.run(X_test)
    else:
        y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred)
    rnmse = rmse / (np.max(y_test) - np.min(y_test))

    results.append([name, rnmse, rmse, mae, mape, r2])

columns = ["Model", "RNMSE", "RMSE", "MAE", "MAPE", "R2"]
results_df = pd.DataFrame(results, columns=columns)
print(results_df.to_string(index=False))
