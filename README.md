# Customer Lifetime Value Prediction and Segmentation Model
Python Version License: MIT

## Overview
This project provides a robust, data-driven customer lifetime value (CLV) prediction and segmentation model designed to analyze and quantify the long-term value of e-commerce customers. By predicting future revenue generation and performing RFM (Recency, Frequency, Monetary) clustering, this solution empowers marketing teams to prioritize retention strategies, optimize market spend, and increase profitability among high-value segments.

## Architecture and Project Structure
### Repository Layout
```text
capstone-project/
├── data/                      # Raw and processed datasets (UCI Online Retail Dataset)
├── notebooks/                 # Exploratory and experimental notebooks
│   ├── eda.ipynb              # Exploratory Data Analysis & visual insights
│   ├── feature_engineering.ipynb  # Feature extraction & RFM workflows
│   └── modeling.ipynb         # Model training, hyperparameter tuning & SHAP
├── src/                       # Production-grade source code
│   ├── data_pipeline.py       # Data cleaning and processing
│   └── training_pipeline.py   # Automated end-to-end model training script
├── model/                     # Model artifacts natively serialized
│   └── trained_model.pkl      # Production-ready trained best model
├── presentation/              # Final presentation and reporting files
├── app.py                     # Interactive Streamlit dashboard
└── README.md                  # Project documentation
```

### Dataset Semantics
The analytical dataset comprises transactional records capturing customer purchasing behavior:
* **Customer_ID**: Unique identifier for each client.
* **Recency**: Days since the customer's last purchase.
* **Frequency**: Total number of unique transactions/invoices.
* **Monetary**: Total spend across all transactions.
* **Segment**: Clustered category (High Value, Medium Value, Low Value).
* **CLV (Target)**: Predicted future monetary value.

## Quickstart & Installation
Follow these instructions to set up the environment locally.

### 1. Environment Setup
**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

**macOS/Linux (Bash):**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 2. Install Dependencies
Install the required data science and machine learning packages:
```bash
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost shap jupyter streamlit
```

## Execution Guide
### Option 1: Automated ML Pipeline (Recommended for Production)
Execute the end-to-end training pipeline via the command line. This script automatically performs data loading, processing, model training, evaluation, and serialization.

```bash
# Run from the project root directory
python src/training_pipeline.py
```
**Expected Output:**
* Generates processed data in the `data/` directory.
* Serializes the trained regression model to `model/trained_model.pkl`.
* Prints evaluation metrics ($R^2$, RMSE) to standard output.

### Option 2: Interactive Analysis (Jupyter Notebooks)
For exploratory data analysis, visual insights, and interactive hyperparameter tuning:
```bash
jupyter notebook
```
Navigate to the `notebooks/` directory and execute the notebooks in the following logical sequence:
1. `eda.ipynb`
2. `feature_engineering.ipynb`
3. `modeling.ipynb`

### Option 3: Interactive Dashboard
To launch the interactive application for business users and "What-If" simulations:
```bash
streamlit run app.py
```

## Model Evaluation Metrics
The modeling pipeline evaluated Linear Regression against non-linear tree-based approaches (Random Forest, XGBoost) for predicting continuous monetary value.

* **Linear Regression:** Selected as the optimal baseline model ($R^2$: 0.34, RMSE: 8199) due to its robustness against the extreme outliers inherent in the monetary spending dataset.
* **Tree-based Models:** Random Forest and XGBoost failed to generalize to the test set ($R^2$: 0.00, RMSE: 10157-10608), over-fitting heavily on the training data.

The pipeline calculates the following key metrics on a holdout test set:
* **R-Squared ($R^2$):** Proportion of the variance in the dependent variable that is predictable.
* **RMSE (Root Mean Squared Error):** Standard deviation of the prediction errors. 

## Strategic Business Insights & SHAP Explainability
Leveraging regression coefficients and feature importance, the model provides interpretable insights into customer value drivers.

### Segmentation Discoveries
* **High-Value Concentration:** Successfully split the customer base of 4,338 unique clients, identifying an elite group of 26 extreme `High Value` customers generating massively outsized returns.
* **Frequency Dominance:** `Frequency` (number of unique invoices) was found to be drastically more important than `Recency` in determining final monetary value.

### Strategic Recommendations
* **Fund VIP Retention:** Dedicate specialized account management and premium loyalty incentives to the 26 extreme High-Value customers to protect the core revenue base.
* **Optimize for Frequency:** Since purchasing frequency is the strongest driver of CLV, marketing campaigns should focus on encouraging repeat purchases (e.g., automated replenishment emails, loyalty points for multiple orders) rather than pushing single large-ticket transactions.
* **Reframe Reactivation KPIs:** Target Medium and Low-value segments nearing churn with aggressive "win-back" campaigns, measuring success by the reactivation of their purchasing frequency.
