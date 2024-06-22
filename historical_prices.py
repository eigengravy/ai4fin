import os
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":

    files = os.listdir("./data/")

    for file in files:
        data = pd.read_csv("./data/" + file)
        data.rename(
            columns={"Reported Date": "Date", "Modal Price (Rs./Quintal)": "Price"},
            inplace=True,
        )
        data = data[["Date", "Price"]]
        data["Date"] = pd.to_datetime(data["Date"])

        data['Price'] = pd.to_numeric(data['Price'], errors='coerce')

        #Drop any rows with NaN values in 'Price'
        data.dropna(subset=['Price'], inplace=True)

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(data['Date'], data['Price'])

        # Set the title and labels
        commodity_name, commodity_state = file.split('_')
        plt.title(commodity_name.upper())
        plt.xlabel('Date')
        plt.ylabel('Price')

        # Rotate date labels for better readability
        plt.xticks(rotation=45)

        # Save the plot as an image
        plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
        plt.savefig('./historical_prices/' + commodity_name + '.png')

        plt.show()