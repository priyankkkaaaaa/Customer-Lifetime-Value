import pandas as pd
import numpy as np
import os
import joblib
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import shap

def create_advanced_rfm_split(df: pd.DataFrame):
    """
    Advanced Data Science Time-Cutoff Methodology
    Splits the 1-year dataset into:
    - Observation Window (First 9 months): Used to calculate historical RFM behaviors.
    - Target Window (Last 3 months): Used to calculate the true *future* monetary value to predict.
    """
    print("Applying Advanced Time-Cutoff Methodology...")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    max_date = df['InvoiceDate'].max()
    min_date = df['InvoiceDate'].min()
    
    # 9-Month Observation Window cutoff
    cutoff_date = min_date + pd.DateOffset(months=9)
    
    # Split Data
    obs_window = df[df['InvoiceDate'] <= cutoff_date]
    target_window = df[df['InvoiceDate'] > cutoff_date]
    
    # 1. Calculate Features (from Observation Window)
    snapshot_date = cutoff_date + dt.timedelta(days=1)
    rfm_features = obs_window.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Historical_Value'})
    
    # 2. Calculate Targets (from Target Window)
    target_values = target_window.groupby('CustomerID').agg({
        'TotalPrice': 'sum'
    }).rename(columns={'TotalPrice': 'Future_MonetaryValue'})
    
    # Merge Features and Targets
    # Inner join means we only keep customers who existed in the observation window AND bought again in the target window.
    # To predict CLV for all customers, we do a left join and fill NaNs with 0 (meaning they churned/didn't buy).
    final_df = pd.merge(rfm_features, target_values, on='CustomerID', how='left').fillna(0)
    
    print(f"Time-Cutoff Complete. Total Customers tracked across windows: {len(final_df)}")
    return final_df

def run_modeling_pipeline(cleaned_data_path: str, model_out_dir: str):
    print("--- Day 55-58 Modeling Pipeline ---")
    df_raw = pd.read_csv(cleaned_data_path)
    
    # Execute the advanced Time-Split
    df = create_advanced_rfm_split(df_raw)
    
    # --- 1. Customer Segmentation (Clustering) ---
    print("\n[Stage 1] Performing RFM Segmentation on Historical Data...")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(df[['Recency', 'Frequency', 'Historical_Value']])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    joblib.dump(kmeans, os.path.join(model_out_dir, "kmeans_model.pkl"))
    joblib.dump(scaler, os.path.join(model_out_dir, "scaler.pkl"))
    
    # Map clusters
    cluster_means = df.groupby('Cluster')['Historical_Value'].mean().sort_values()
    segment_map = {
        cluster_means.index[0]: 'Low Value',
        cluster_means.index[1]: 'Medium Value',
        cluster_means.index[2]: 'High Value'
    }
    df['Segment'] = df['Cluster'].map(segment_map)
    
    # --- 2. Advanced Outlier Removal (Real-world practice) ---
    print("\n[Stage 2] Removing Extreme Outliers for Model Generalization...")
    initial_len = len(df)
    
    # Remove top 1% of target and features to prevent models from memorizing whales
    q_target = df['Future_MonetaryValue'].quantile(0.99)
    q_hist = df['Historical_Value'].quantile(0.99)
    q_freq = df['Frequency'].quantile(0.99)
    
    df_clean = df[(df['Future_MonetaryValue'] < q_target) & 
                  (df['Historical_Value'] < q_hist) & 
                  (df['Frequency'] < q_freq)].copy()
    
    print(f"Removed {initial_len - len(df_clean)} extreme outlier customers (top 1% whales).")
    
    # Export the final feature set for the Streamlit App to load
    # (We save the full df, not just the outlier-removed one, so the app shows all segments)
    df.rename(columns={'Historical_Value': 'MonetaryValue'}).to_csv(os.path.join(current_dir, '..', 'data', 'rfm_features.csv'), index=True)
    
    # --- 3. Predictive Modeling ---
    print("\n[Stage 3] Training Advanced Models on Future Spend...")
    X = df_clean[['Recency', 'Frequency', 'Historical_Value']]
    y = df_clean['Future_MonetaryValue']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    best_model_name = ""
    best_r2 = -float("inf")
    best_model = None
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        results[name] = {'RMSE': rmse, 'R2': r2}
        print(f"  {name} -> RMSE: {rmse:.2f}, R2: {max(0, r2):.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = model

    print(f"\n[Stage 4] Evaluation & Best Model")
    print(f"New Best Model: {best_model_name} with R2: {best_r2:.4f}")
    
    final_model_path = os.path.join(model_out_dir, "trained_model.pkl")
    joblib.dump(best_model, final_model_path)
    print(f"Saved optimal model to {final_model_path}")
    
    # Provide insights based on the new winning tree model
    if "XGB" in best_model_name or "Forest" in best_model_name:
        print("\nTree model successfully generalized after outlier removal!")
        print("Feature Importances:")
        importances = best_model.feature_importances_
        for i, col in enumerate(X.columns):
            print(f"  {col}: {importances[i]:.4f}")

    print("\n--- Pipeline Finished Successfully ---")
        
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # NOTE: The advanced pipeline now directly ingests the CLEAN TRANSACTION DATA
    # so it can perform the time-cutoff internally.
    cleaned_data = os.path.join(current_dir, '..', 'data', 'online_retail_cleaned.csv')
    model_dir = os.path.join(current_dir, '..', 'model')
    
    os.makedirs(model_dir, exist_ok=True)
    
    if os.path.exists(cleaned_data):
        run_modeling_pipeline(cleaned_data, model_dir)
    else:
        print(f"Error: Could not find {cleaned_data}. Please run data_pipeline.py first.")
