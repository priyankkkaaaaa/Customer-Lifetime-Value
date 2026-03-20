import nbformat as nbf
import os

def create_eda():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Day 52 - Data Understanding (EDA)\n\n## 1. Data Loading"),
        nbf.v4.new_code_cell("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Load the dataset\ndf = pd.read_csv('../data/online_retail.csv')\ndf.head()"),
        nbf.v4.new_markdown_cell("## 2. Data Dictionary\n*   **InvoiceNo**: Invoice number. A 6-digit integral number uniquely assigned to each transaction.\n*   **StockCode**: Product (item) code.\n*   **Description**: Product (item) name.\n*   **Quantity**: The quantities of each product per transaction.\n*   **InvoiceDate**: Invice Date and time.\n*   **UnitPrice**: Unit price.\n*   **CustomerID**: Customer number.\n*   **Country**: Country name."),
        nbf.v4.new_code_cell("df.info()"),
        nbf.v4.new_markdown_cell("## 3. Missing Value Analysis"),
        nbf.v4.new_code_cell("# Check for missing values\nmissing_values = df.isnull().sum()\nmissing_percentages = (missing_values / len(df)) * 100\npd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentages}).sort_values('Percentage', ascending=False)"),
        nbf.v4.new_markdown_cell("## 4. Initial Insights & Data Cleaning Prep"),
        nbf.v4.new_code_cell("df.describe()"),
        nbf.v4.new_code_cell("# Top 10 Countries by transaction count\nplt.figure(figsize=(12,6))\nsns.countplot(data=df, y='Country', order=df['Country'].value_counts().iloc[:10].index)\nplt.title('Top 10 Countries by Transaction Count')\nplt.ylabel('')\nplt.show()"),
    ]
    with open('../notebooks/eda.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

def create_feature_engineering():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Day 54 - Feature Engineering (RFM Metrics)"),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport datetime as dt\n\n# Load cleaned data (or clean it here)\ndf = pd.read_csv('../data/online_retail.csv')\n\n# Quick clean\ndf = df.dropna(subset=['CustomerID'])\ndf = df[df['Quantity'] > 0]\ndf['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])\ndf['TotalPrice'] = df['Quantity'] * df['UnitPrice']"),
        nbf.v4.new_markdown_cell("## RFM Calculation\nRecency, Frequency, Monetary value per CustomerID."),
        nbf.v4.new_code_cell("snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)\n\nrfm = df.groupby('CustomerID').agg({\n    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,\n    'InvoiceNo': 'nunique',\n    'TotalPrice': 'sum'\n})\nrfm.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'MonetaryValue'}, inplace=True)\nrfm.head()"),
    ]
    with open('../notebooks/feature_engineering.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

def create_modeling():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Day 55-58 - Modeling, Evaluation & Explainability"),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nfrom sklearn.cluster import KMeans\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn.metrics import mean_squared_error, r2_score\nimport xgboost as xgb\nimport shap"),
    ]
    with open('../notebooks/modeling.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    os.makedirs('../notebooks', exist_ok=True)
    create_eda()
    create_feature_engineering()
    create_modeling()
    print("Notebooks successfully generated.")
