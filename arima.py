import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Preprocess data
dataset_ex_df = pd.read_csv("../Wheat daily price__UP_Agra.csv")
dataset_ex_df = dataset_ex_df.reset_index()
dataset_ex_df['Date'] = pd.to_datetime(dataset_ex_df['Date'])
dataset_ex_df.set_index('Date', inplace=True)
dataset_ex_df = dataset_ex_df['Prices'].to_frame()


from pmdarima.arima import auto_arima

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

# Auto ARIMA to select optimal ARIMA parameters
model = auto_arima(dataset_ex_df['Prices'], seasonal=False, trace=True)
print(model.summary())

from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Define the ARIMA model
def arima_forecast(history):
    # Fit the model
    model = ARIMA(history, order=(0,1,1))
    model_fit = model.fit()
    
    # Make the prediction
    output = model_fit.forecast()
    yhat = output[0]
    return yhat

# Split data into train and test sets
X = dataset_ex_df.values
size = int(len(X) * 0.8)
train, test = X[0:size], X[size:len(X)]

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

# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 6), dpi=100)
# plt.plot(dataset_ex_df.iloc[size:,:].index, test, label='Real')
# plt.plot(dataset_ex_df.iloc[size:,:].index, predictions, color='red', label='Predicted')
# plt.title('ARIMA Predictions vs Actual Values')
# plt.xlabel('Date')
# plt.ylabel('Stock Price')
# plt.legend()
# plt.show()

# print(test)
# print(predictions)

rmse, rnmse, mae, mape, r_squared = calculate_metrics(test, predictions)
print(f"RMSE: {rmse}")
print(f"RNMSE: {rnmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}%")
print(f"R-squared: {r_squared}")