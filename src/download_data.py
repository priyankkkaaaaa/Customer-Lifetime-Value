import pandas as pd
import requests
import os

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
raw_xl_path = os.path.join(data_dir, "Online_Retail.xlsx")
csv_path = os.path.join(data_dir, "online_retail.csv")

if not os.path.exists(csv_path):
    print("Downloading Online Retail dataset...")
    # Using pandas directly to read the Excel file from URL and save to CSV
    try:
        # It's better to download the file directly first
        response = requests.get(url)
        with open(raw_xl_path, 'wb') as f:
            f.write(response.content)
        
        print("Reading Excel file and converting to CSV...")
        df = pd.read_excel(raw_xl_path)
        df.to_csv(csv_path, index=False)
        print(f"Data successfully saved to {csv_path}")
        print(f"Dataset shape: {df.shape}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure openpyxl is installed: pip install openpyxl")
else:
    print(f"Dataset already exists at {csv_path}")

