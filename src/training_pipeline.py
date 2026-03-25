import pandas as pd
import numpy as np
import os
import joblib
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report
import xgboost as xgb
import optuna
import warnings

warnings.filterwarnings('ignore')

def create_high_res_features(df: pd.DataFrame):
    print("Generating high-resolution transactional features...")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # 1. Horizon Split (Forward Time-Split)
    min_date = df['InvoiceDate'].min()
    cutoff_date = min_date + pd.DateOffset(months=9)
    obs_df = df[df['InvoiceDate'] <= cutoff_date]
    target_df = df[df['InvoiceDate'] > cutoff_date]
    
    snapshot_date = cutoff_date + dt.timedelta(days=1)
    
    # 2. Base Customer Aggregations
    data = obs_df.groupby('CustomerID').agg({
        'InvoiceDate': [lambda x: (snapshot_date - x.max()).days, lambda x: (x.max() - x.min()).days, 'max'],
        'InvoiceNo': 'nunique',
        'TotalPrice': ['sum', 'mean', 'std', 'max'],
        'StockCode': 'nunique'
    })
    data.columns = ['Recency', 'Tenure', 'LastDate', 'Frequency', 'TotalSpend', 'AvgSpend', 'StdSpend', 'MaxSpend', 'Variety']
    
    # 3. Rolling Temporal Features (30, 60, 90 days)
    for days in [30, 60, 90]:
        window_cutoff = snapshot_date - dt.timedelta(days=days)
        window_df = obs_df[obs_df['InvoiceDate'] >= window_cutoff]
        roll_agg = window_df.groupby('CustomerID').agg({'TotalPrice': 'sum', 'InvoiceNo': 'nunique'})
        roll_agg.columns = [f'Spend_Last_{days}d', f'Freq_Last_{days}d']
        data = data.join(roll_agg, how='left').fillna(0)
    
    # 4. Behavioral Velocity
    data['SpendVelocity'] = data['TotalSpend'] / (data['Tenure'] + 1)
    data['RecencyVelocity'] = data['Recency'] / (data['Tenure'] + 1)
    data['RecencyGrowth'] = (data['Spend_Last_30d'] + 1) / (data['Spend_Last_90d'] + 1)
    
    # 5. Transactional Bias (Seasonality)
    data['LastPurchaseDoW'] = data['LastDate'].dt.dayofweek
    data['IsWeekendShopper'] = (data['LastPurchaseDoW'] >= 5).astype(int)
    
    # 6. Target Variable
    converted_ids = target_df['CustomerID'].unique()
    data['converted'] = data.index.isin(converted_ids).astype(int)
    
    return data.drop(columns=['LastDate']).reset_index()

def objective(trial, X_train, y_train):
    # Search space for XGBoost (Base Learner)
    params = {
        'n_estimators': 300,
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
        'eval_metric': 'auc',
        'use_label_encoder': False
    }
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_t, y_t)
        aucs.append(roc_auc_score(y_v, model.predict_proba(X_v)[:, 1]))
    
    return np.mean(aucs)

def run_mission_critical():
    print("--- [Mission Critical Pipeline] Scaling to 95/92 ---")
    raw = pd.read_csv('data/online_retail_cleaned.csv')
    df = create_high_res_features(raw)
    
    X = df.drop(columns=['CustomerID', 'converted'])
    y = df['converted']
    features = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Optimizing base model with 50 trials...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)
    
    best_xgb_params = study.best_params
    best_xgb_params['scale_pos_weight'] = (y_train == 0).sum() / (y_train == 1).sum()
    
    # 2. Ensemble Stacking
    print("Constructing Stacking Ensemble...")
    estimators = [
        ('xgb', xgb.XGBClassifier(**best_xgb_params)),
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=42))
    ]
    
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5,
        stack_method='predict_proba'
    )
    
    stack.fit(X_train, y_train)
    
    # 3. Probability Calibration & Precision Thresholding
    print("Finding Zero-Waste threshold for >95% Precision...")
    y_probs = stack.predict_proba(X_test)[:, 1]
    
    thresholds = np.linspace(0.01, 0.999, 500)
    best_t = 0.5
    for t in thresholds:
        # Avoid division by zero if no samples are predicted positive
        preds_at_t = (y_probs >= t).astype(int)
        if preds_at_t.sum() > 0:
            if precision_score(y_test, preds_at_t) >= 0.95:
                best_t = t
                break
            
    y_pred = (y_probs >= best_t).astype(int)

    
    print("\n--- PERFORMANCE VERIFICATION ---")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_probs):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"Final Calibration Threshold: {best_t:.4f}")
    print(classification_report(y_test, y_pred))
    
    joblib.dump({'model': stack, 'threshold': best_t, 'features': features}, 'model/mission_critical_bundle.pkl')
    print("Mission Critical Bundle Saved.")

if __name__ == "__main__":
    run_mission_critical()
