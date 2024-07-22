import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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


def hybrid_arima_ann(train_data, test_data, arima_order=(1, 1, 1), nn_units=[64, 32]):
    # Fit ARIMA model
    arima_model = ARIMA(train_data, order=arima_order)
    arima_result = arima_model.fit()

    # Generate ARIMA predictions for train and test data
    arima_train_pred = arima_result.predict(start=0, end=len(train_data) - 1)
    arima_test_pred = arima_result.forecast(steps=len(test_data))

    # Calculate residuals
    train_residuals = train_data - arima_train_pred

    # Prepare data for ANN
    X_train = np.array(train_residuals[:-1]).reshape(-1, 1)
    y_train = np.array(train_residuals[1:]).reshape(-1, 1)

    # Create and train ANN model
    ann_model = Sequential()
    for units in nn_units:
        ann_model.add(Dense(units, activation="relu"))
    ann_model.add(Dense(1))
    ann_model.compile(optimizer="adam", loss="mse")
    ann_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    # Make final predictions
    arima_predictions = arima_test_pred
    last_residual = train_residuals[-1]
    ann_predictions = []

    for arima_pred in arima_predictions:
        ann_pred = ann_model.predict(np.array([[last_residual]]))
        hybrid_pred = arima_pred + ann_pred[0][0]
        ann_predictions.append(hybrid_pred)
        last_residual = hybrid_pred - arima_pred

    return np.array(ann_predictions)


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


# python statement to run with all the files
# python3 price_forecast.py Turmeric_Maharashtra.csv Mustard_Rajasthan.csv Paddy_Chattisgarh.csv Greengram_UP.csv Cotton_Gujarat.csv DryChillies_AP.csv KabuliChana_MP.csv Sesamum_Gujarat.csv Cumin_Gujarat.csv Soybean_MP.csv Potato_UP.csv Groundnut_Guj.csv Jowar_Maharashtra.csv Lentil_MP.csv Maize_Chattisgarh.csv Arhar_Maharashtra.csv Onion_Maharashtra.csv Tomato_UP.csv Bengalgram_MP.csv Wheat_UP.csv Ragi_Karnataka.csv Coriander_Gujarat.csv Bajra_Guj.csv


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("Usage: python price_forecast.py file1.csv file2.csv ...")
        sys.exit(1)

    for file in sys.argv[1:]:
        series_values_dict = {}

        data = pd.read_csv(file)
        data.rename(
            columns={"Reported Date": "Date", "Modal Price (Rs./Quintal)": "Price"},
            inplace=True,
        )
        data = data[["Date", "Price"]]
        data["Date"] = pd.to_datetime(data["Date"])

        print(data.head())

        train_data, test_data = train_test_split(data, test_size=0.8, shuffle=False)

        def create_sequences(data, window_size):
            X, y = [], []
            for i in range(window_size, len(data)):
                X.append(data[i - window_size : i])
                y.append(data[i])
            return np.array(X), np.array(y)

        window_size = 20
        X_train, y_train = create_sequences(
            train_data["Price"].values, window_size=window_size
        )
        X_test, y_test = create_sequences(
            test_data["Price"].values, window_size=window_size
        )

        print(y_test[:5])
        series_values_dict["Actual"] = y_test

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

        # Inside the main loop, after creating other models
        hybrid_predictions = hybrid_arima_ann(
            train_data["Price"].values,
            test_data["Price"].values,
        )[window_size:]

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

            series_values_dict[name] = y_pred.ravel()

            results.append([name, rnmse, rmse, mae, mape, r2])

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
        train, test = y_train.tolist(), y_test.tolist()

        # Walk-forward validation
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            # Generate a prediction
            yhat = arima_forecast(history)
            predictions.append(yhat)
            # Add the predicted value to the training set
            obs = test[t]
            history.append(obs)

        series_values_dict["ARIMA Actual"] = test
        series_values_dict["ARIMA"] = predictions
        series_values_dict["Hybrid ARIMA-ANN"] = hybrid_predictions

        rmse_arima = np.sqrt(mean_squared_error(test, predictions))
        mae_arima = mean_absolute_error(test, predictions)
        mape_arima = mean_absolute_percentage_error(test, predictions) * 100
        r_squared_arima = r2_score(test, predictions)
        rnmse_arima = rmse_arima / (np.max(predictions) - np.min(predictions))

        results.append(
            ["ARIMA", rnmse_arima, rmse_arima, mae_arima, mape_arima, r_squared_arima]
        )

        # print("DEBUG`")
        # print(len(test))
        # print(hybrid_predictions.shape)
        # print(hybrid_predictions)
        # print(test)
        rmse_hybrid = np.sqrt(mean_squared_error(test, hybrid_predictions))
        mae_hybrid = mean_absolute_error(test, hybrid_predictions)
        mape_hybrid = mean_absolute_percentage_error(test, hybrid_predictions) * 100
        r_squared_hybrid = r2_score(test, hybrid_predictions)
        rnmse_hybrid = rmse_hybrid / (np.max(predictions) - np.min(hybrid_predictions))

        results.append(
            [
                "Hybrid ARIMA-ANN",
                rnmse_hybrid,
                rmse_hybrid,
                mae_hybrid,
                mape_hybrid,
                r_squared_hybrid,
            ]
        )

        columns = ["Model", "RNMSE", "RMSE", "MAE", "MAPE", "R2"]
        results_df = pd.DataFrame(results, columns=columns)
        print(results_df.to_string(index=False))
        os.makedirs("outputs", exist_ok=True)
        filename = file.split("/")[-1].split(".")[0]
        results_df.to_csv(f"outputs/{filename}_results.csv", index=False)

        print(series_values_dict)
        series_values_df = pd.DataFrame(series_values_dict)
        series_values_df.to_csv("series_values/" + file, index=False)
