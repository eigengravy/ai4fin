import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


files = os.listdir("./series_values/")

for file in files:
    if file == '.DS_Store':
        continue

    df = pd.read_csv("./series_values/" + file)

    print(df)

    commodity, state = file.split('_')
    commodity.upper()

    # Plotting all columns
    plt.figure(figsize=(50, 40))  # Set a large size for the plot

    for column in df.columns:
        if column == 'ESN' or column == "ARIMA Actual":
            continue
        plt.plot(df.index, df[column], label=column)

    # Adding labels and title
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title(commodity)
    plt.legend(prop={'size': 30})

    # Saving the plot
    plt.savefig('./prediction_images/plot_no_esn_' + commodity + '.png')

    # Display the plot
    # plt.show()

