import os
import pandas as pd

files = os.listdir("./outputs 2/")
    
titles = ["Commodity", "MLP", "GRU", "LSTM", "RNN", "XGBoost", "Linear Regression", "SVR", "ESN", "ARIMA"]
rmse = []
rnmse = []
mae = []
mape = []

for file in files:
    if file == ".DS_Store":
        continue
    df = pd.read_csv("./outputs 2/" + file)
    commodity = file[:-12]
    rmse_col = [commodity] + df.iloc[:, 2].values.tolist()
    rnmse_col = [commodity] + df.iloc[:, 1].values.tolist()
    mae_col = [commodity] + df.iloc[:, 3].values.tolist()
    mape_col = [commodity] + df.iloc[:, 4].values.tolist()

    
    rmse.append(rmse_col)
    rnmse.append(rnmse_col)
    mae.append(mae_col)
    mape.append(mape_col)

df = pd.DataFrame(rmse, columns=titles)
df.to_csv("./formatted_results/rmse.csv", index=False)

df = pd.DataFrame(rnmse, columns=titles)
df.to_csv("./formatted_results/rnmse.csv", index=False)

df = pd.DataFrame(mae, columns=titles)
df.to_csv("./formatted_results/mae.csv", index=False)

df = pd.DataFrame(mape, columns=titles)
df.to_csv("./formatted_results/mape.csv", index=False)



    
        
    