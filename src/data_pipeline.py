import pandas as pd
import os

def clean_data(input_path: str, output_path: str):
    """
    Day 53 - Data Cleaning Pipeline
    Loads raw data, handles missing values, removes outliers (cancellations),
    and performs initial data transformations.
    """
    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)
    initial_shape = df.shape
    
    # 1. Handle Missing Values
    # CustomerID is critical for CLV and RFM analysis. We must drop rows without it.
    df.dropna(subset=['CustomerID'], inplace=True)
    
    # 2. Outlier Handling / Data Cleaning
    # Remove cancelled orders (Quantity < 0) or erroneous prices (UnitPrice <= 0)
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    # 3. Data Transformation
    # Ensure proper data types
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str) # Convert to string to treat as categorical
    
    # 4. Feature Engineering Prep
    # Create TotalPrice column
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    final_shape = df.shape
    print(f"Cleaning complete. Rows dropped: {initial_shape[0] - final_shape[0]}")
    print(f"Saved cleaned dataset to {output_path}")
    
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, '..', 'data', 'online_retail.csv')
    output_file = os.path.join(current_dir, '..', 'data', 'online_retail_cleaned.csv')
    
    if os.path.exists(input_file):
        clean_data(input_file, output_file)
    else:
        print(f"Error: Could not find {input_file}. Please run download_data.py first.")
