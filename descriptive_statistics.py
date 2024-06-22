import os
import pandas as pd
import sys
from scipy.stats import skew, kurtosis, jarque_bera

#python statement to run with all the files
#python3 descriptive_statistics.py Turmeric_Maharashtra.csv Mustard_Rajasthan.csv Paddy_Chattisgarh.csv Greengram_UP.csv Cotton_Gujarat.csv DryChillies_AP.csv KabuliChana_MP.csv Sesamum_Gujarat.csv Cumin_Gujarat.csv Soybean_MP.csv Potato_UP.csv Groundnut_Guj.csv Jowar_Maharashtra.csv Lentil_MP.csv Maize_Chattisgarh.csv Arhar_Maharashtra.csv Onion_Maharashtra.csv Tomato_UP.csv Bengalgram_MP.csv Wheat_UP.csv Ragi_Karnataka.csv Coriander_Gujarat.csv Bajra_Guj.csv


if __name__ == '__main__':
    columns = ["Commodity", "State", "Mean", "Median", "Maximum", "Minimum", "Std Dev", "CV", "Skewness", "Kurtoises", "Jarque-Bera", "Probability"]
    results = [columns]

    if len(sys.argv) < 1:
        print("Usage: python descriptive_statistics.py file1.csv file2.csv ...")
        sys.exit(1)

    for file in sys.argv[1:]:
        data = pd.read_csv("./data/" + file)
        data.rename(
            columns={"Reported Date": "Date", "Modal Price (Rs./Quintal)": "Price"},
            inplace=True,
        )
        data = data[["Date", "Price"]]
        data["Date"] = pd.to_datetime(data["Date"])

        data['Price'] = pd.to_numeric(data['Price'], errors='coerce')

        # Drop any rows with NaN values in 'Price'
        data.dropna(subset=['Price'], inplace=True)

        # Calculations
        mean_price = data['Price'].mean()
        median_price = data['Price'].median()
        max_price = data['Price'].max()
        min_price = data['Price'].min()
        std_dev_price = data['Price'].std()
        cv_price = std_dev_price / mean_price
        skewness_price = skew(data['Price'])
        kurtosis_price = kurtosis(data['Price'])
        jb_stat, jb_p_value = jarque_bera(data['Price'])

        commodity_name, commodity_state = file.split('_')

        commodity_result = [commodity_name, commodity_state[:-4], mean_price, median_price, max_price, min_price, std_dev_price, cv_price, skewness_price, kurtosis_price, jb_stat, jb_p_value]
        results.append(commodity_result)

        # Display results
        print(f"Mean: {mean_price}")
        print(f"Median: {median_price}")
        print(f"Maximum: {max_price}")
        print(f"Minimum: {min_price}")
        print(f"Standard Deviation: {std_dev_price}")
        print(f"Coefficient of Variation: {cv_price}")
        print(f"Skewness: {skewness_price}")
        print(f"Kurtosis: {kurtosis_price}")
        print(f"Jarque-Bera Statistic: {jb_stat}")
        print(f"Jarque-Bera p-value: {jb_p_value}")

    df = pd.DataFrame(results)
    df.to_csv("descriptive_statistics.csv", index=False, header=False)



