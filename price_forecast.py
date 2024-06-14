import os
import sys
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

from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA


def calculate_metrics(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)

    # RMSE
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # Relative RMSE (RRMSE)
    rnmse = rmse / (np.max(actual) - np.min(actual))

    # MAE
    mae = mean_absolute_error(actual, predicted)

    # MAPE
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    # R-squared
    r_squared = r2_score(actual, predicted)

    return rmse, rnmse, mae, mape, r_squared


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("Usage: python price_forecast.py file1.csv file2.csv ...")
        sys.exit(1)

    for file in sys.argv[1:]:
        data = pd.read_csv(file)
        data.rename(
            columns={"Reported Date": "Date", "Modal Price (Rs./Quintal)": "Price"},
            inplace=True,
        )
        data = data[["Date", "Price"]]
        data["Date"] = pd.to_datetime(data["Date"])

        print(data.head())

        train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

        scaler = MinMaxScaler()
        train_data["Price"] = scaler.fit_transform(
            train_data["Price"].values.reshape(-1, 1)
        ).ravel()
        test_data["Price"] = scaler.transform(
            test_data["Price"].values.reshape(-1, 1)
        ).ravel()

        def create_sequences(data, window_size):
            X, y = [], []
            for i in range(window_size, len(data)):
                X.append(data[i - window_size : i])
                y.append(data[i])
            return np.array(X), np.array(y)

        window_size = len(test_data) // 10
        X_train, y_train = create_sequences(
            train_data["Price"].values, window_size=window_size
        )
        X_test, y_test = create_sequences(
            test_data["Price"].values, window_size=window_size
        )

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
        reservoir = Reservoir(units=5000, lr=0.3, sr=0.8)
        ridge = Ridge(ridge=1e-7)
        esn = reservoir >> ridge
        esn.fit(X_train, y_train.reshape(-1, 1))
        models.append(("ESN", esn))

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

        # Running ARIMA
        dataset_ex_df = pd.read_csv(file)
        dataset_ex_df.rename(
            columns={"Reported Date": "Date", "Modal Price (Rs./Quintal)": "Price"},
            inplace=True,
        )
        dataset_ex_df = dataset_ex_df[["Date", "Price"]]
        dataset_ex_df["Date"] = pd.to_datetime(dataset_ex_df["Date"])
        dataset_ex_df = dataset_ex_df.reset_index()
        dataset_ex_df["Date"] = pd.to_datetime(dataset_ex_df["Date"])
        dataset_ex_df.set_index("Date", inplace=True)
        dataset_ex_df = dataset_ex_df["Price"].to_frame()
        model = auto_arima(dataset_ex_df["Price"], seasonal=False, trace=True)

        # Define the ARIMA model
        def arima_forecast(history):
            # Fit the model
            model = ARIMA(history, order=(0, 1, 1))
            model_fit = model.fit()

            # Make the prediction
            output = model_fit.forecast()
            yhat = output[0]
            return yhat

        # Split data into train and test sets
        X = dataset_ex_df.values
        size = int(len(X) * 0.8)
        train, test = list(X[0:size]), list(X[size : len(X)])

        def min_max_scale_list(data_list):
            # Check if the input is a list
            if not isinstance(data_list, list):
                raise ValueError("Input must be a list")
            
            # Reshape the data to fit the scaler's requirements
            data_array = np.array(data_list).reshape(-1, 1)
            
            # Initialize the scaler
            scaler = MinMaxScaler()
            
            # Fit and transform the data
            scaled_data = scaler.fit_transform(data_array).ravel()
            
            # Return the scaled data as a list
            return scaled_data.tolist()

        train = min_max_scale_list(train)
        test = min_max_scale_list(test)

        # Walk-forward validation
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            # Generate a prediction
            yhat = arima_forecast(history)
            predictions.append(yhat)
            # Add the predicted value to the training set
            # obs = test[t]
            # history.append(obs)

        rmse_arima, rnmse_arima, mae_arima, mape_arima, r_squared_arima = (
            calculate_metrics(test, predictions)
        )

        results.append(
            ["ARIMA", rnmse_arima, rmse_arima, mae_arima, mape_arima, r_squared_arima]
        )

        columns = ["Model", "RNMSE", "RMSE", "MAE", "MAPE", "R2"]
        results_df = pd.DataFrame(results, columns=columns)
        print(results_df.to_string(index=False))
        os.makedirs("outputs", exist_ok=True)
        filename = file.split("/")[-1].split(".")[0]
        results_df.to_csv(f"outputs/{filename}_results.csv", index=False)
