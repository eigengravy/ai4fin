import os
import sys
import pandas as pd

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("Usage: python split_xlsx.py <filename.xlsx>")

    wb = pd.ExcelFile(sys.argv[1])
    sheets = wb.sheet_names

    for sheet in sheets:
        df = wb.parse(sheet)
        os.makedirs("data", exist_ok=True)
        df.to_csv(f"data/{sheet}.csv", index=False)
