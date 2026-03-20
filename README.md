# Customer Lifetime Value Prediction and Segmentation

## Capstone Project Overview

### Problem Statement
An e-commerce company wants to identify customers who will generate the highest long-term revenue. Currently, marketing resources are spent uniformly across the customer base, which is inefficient. The company needs to predict the Customer Lifetime Value (CLV) based on historical transactions, perform RFM (Recency, Frequency, Monetary) analysis, and cluster customers into meaningful segments.

### Business Objective
To help the marketing team prioritize retention strategies for highly valuable customers. By identifying who the high-value customers are and predicting their future revenue generation, the company can target personalized campaigns, optimize market spend, and ultimately increase retention among its most profitable segments.

### Success Metrics
1. **Segmentation Quality:** Meaningful customer segments identified through K-Means clustering on RFM variables (High Value, Medium Value, Low Value).
2. **Model Performance:** The predictive model for CLV should achieve an R-squared ($R^2$) score of at least 0.6 and a solid baseline RMSE compared to average customer value.
3. **Actionable Insights:** Clear, data-driven retention strategies synthesized from the model's feature importance and SHAP explanations.

---

## Directory Structure
- `data/`: Contains raw and processed datasets (UCI Online Retail Dataset).
- `notebooks/`: Jupyter notebooks for EDA, Feature Engineering, and Modeling.
- `src/`: Python scripts for data pipelines and training.
- `model/`: Serialized trained models.
- `presentation/`: Final presentation files.

## Reproducing the Project
### 1. Requirements
Ensure you have the required packages: `pandas`, `scikit-learn`, `xgboost`, `shap`, `jupyter`.

### 2. Execution Pipeline
1. Run `python src/download_data.py` to fetch the UCI dataset.
2. Run `python src/generate_notebooks.py` to recreate the notebooks.
3. Run `jupyter nbconvert --to notebook --execute notebooks/eda.ipynb --inplace` (or run manually).
4. Run `python src/data_pipeline.py` to clean the data.
5. Run the feature engineering notebook or execute the python script equivalents.
6. Run `python src/training_pipeline.py` to train models and export `model/trained_model.pkl`.

## Final Evaluation Report (Day 57 & Day 60)
* **Dataset Segmentation:** Successfully split the customer base of 4,338 unique clients into High, Medium, and Low-Value segments using K-Means clustering.
* **Top Metric:** Found 26 extreme `High Value` customers generating high returns.
* **Model Comparison:** 
  * Linear Regression ($R^2$: 0.34, RMSE: 8199)
  * Random Forest ($R^2$: 0.00, RMSE: 10608)
  * XGBoost ($R^2$: 0.00, RMSE: 10157)
* **Conclusion:** Linear Regression was selected as the optimal baseline model because the non-linear tree-based approaches suffered significantly from the extreme outliers inherent in the monetary spending dataset and failed to generalize to the test set.
* **Feature Importance:** Using regression coefficients, `Frequency` (number of unique invoices) was drastically more important than `Recency` in determining final monetary value.
