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
The modeling pipeline evaluated Logistic Regression against tree-based classifiers (Random Forest, XGBoost) to predict rigorous, future window conversions (High-Value Customers vs Churned/Low-Value Customers).

* **XGBoost (Tuned):** Selected as the optimal model after GridSearchCV hyperparameter tuning (`ROC-AUC: ~0.73`, `Accuracy: ~67%`, `Precision: ~68%`). It successfully generalized to the holdout test set, masterfully handling class imbalances.
* **Logistic Regression & Random Forest:** Served as robust baselines, but the gradient boosting mechanism natively handled the complex, non-linear RFM interactions better.

The pipeline calculates the following key classification metrics on a holdout test set:
* **ROC-AUC:** Area under the Receiver Operating Characteristic curve, representing the model's ability to seamlessly distinguish between High-Value returning clients and churned accounts.
* **Precision & Recall:** Evaluates the ratio of true VIP classifications against false-positives and missed opportunities. 

## Strategic Business Insights & SHAP Explainability
Leveraging XGBoost architecture and rigorous SHAP (SHapley Additive exPlanations) values, the model provides interpretable insights into what drives a customer to return and generate massive future value.

### Segmentation Discoveries
* **K-Means Clustering:** Successfully segmented the historical customer base into 3 distinct demographic tiers based on scaled RFM profiles (e.g., Platinum, Gold, Silver).
* **Time-To-Conversion is Critical:** SHAP explainability revealed that the days between a user's first touchpoint and their conversion event natively dictated their likelihood to become a returning VIP.
* **Frequency Dominance:** `Frequency` (number of unique transactions) remains a vastly stronger predictor of future High Value than purely massive single-ticket `MonetaryValue` drops.

### Strategic Recommendations
* **Target the Platinum Cluster:** Dedicate premium loyalty incentives to the K-Means 'Platinum' segment mapped by the algorithm to proactively combat churn in the top future revenue base.
* **Optimize for Frequency:** Since purchasing frequency drives future CLV predictability, marketing campaigns should focus on encouraging repeat purchases (e.g., automated replenishment emails, loyalty points for multiple orders).
* **Strict Predictive Triggers:** Utilize the XGBoost predictive probabilities to trigger automated "win-back" campaigns the exact moment a high-probability customer shows early warning signs of timeline churn.
