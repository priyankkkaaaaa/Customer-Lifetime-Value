import nbformat as nbf
import os

def create_eda():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Customer Lifetime Value - Exploratory Data Analysis (EDA)\n\nIn this notebook, we explore the `raw_data.csv` (Online Retail) dataset to understand customer interactions and their impact on long-term value.\n\n---\n\n## 1. Import Libraries\n\n---"),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')"),
        nbf.v4.new_markdown_cell("## 2. Load Dataset\n\n---"),
        nbf.v4.new_code_cell("try:\n    df = pd.read_csv('../data/online_retail.csv')\n    print('Dataset loaded successfully with shape:', df.shape)\nexcept Exception as e:\n    print('Error loading dataset:', e)"),
        nbf.v4.new_markdown_cell("## 3. Dataset Overview and Missing Values\n\n---"),
        nbf.v4.new_code_cell("missing_values = df.isnull().sum()\nprint('Missing Values:\\n', missing_values[missing_values > 0])\n\n# Drop missing CustomerIDs for analysis\ndf.dropna(subset=['CustomerID'], inplace=True)\ndf = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]\nprint('Cleaned dataset shape:', df.shape)"),
        nbf.v4.new_markdown_cell("## 4. Purchasing Channel & Country Distribution\n\n---"),
        nbf.v4.new_code_cell("country_dist = df['Country'].value_counts().head(10)\nplt.figure(figsize=(10, 5))\nsns.barplot(x=country_dist.values, y=country_dist.index, palette='viridis')\nplt.title('Top 10 Countries by Transaction Count')\nplt.show()"),
        nbf.v4.new_markdown_cell("## 5. Value Distribution per Customer\n\n---"),
        nbf.v4.new_code_cell("df['TotalPrice'] = df['Quantity'] * df['UnitPrice']\ncustomer_value = df.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False)\nprint('Top 5 Customers by Value:\\n', customer_value.head())\n\nplt.figure(figsize=(8, 4))\nsns.histplot(customer_value[customer_value < 5000], bins=50, kde=True)\nplt.title('Distribution of Customer Monetary Value (Clipped < $5000)')\nplt.show()"),
        nbf.v4.new_markdown_cell("## 6. Time-based Interaction Analysis\nLet's see how purchasing behaviors vary by time.\n\n---"),
        nbf.v4.new_code_cell("df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])\ndf['Hour'] = df['InvoiceDate'].dt.hour\nplt.figure(figsize=(8, 4))\nsns.countplot(x='Hour', data=df, palette='magma')\nplt.title('Transactions by Hour of Day')\nplt.show()\n\ndf.to_csv('../data/online_retail_cleaned.csv', index=False)")
    ]
    with open('../notebooks/eda.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

def create_feature_engineering():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Customer Lifetime Value - Feature Engineering\n\nIn this notebook, we transform the raw transactional data into user-level features to prepare for modeling.\n\n---\n\n## 1. Import Libraries and Load Data\n\n---"),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport datetime as dt\n\ntry:\n    df = pd.read_csv('../data/online_retail_cleaned.csv')\n    print('Data loaded.')\nexcept Exception as e:\n    print('Failed to load data:', e)"),
        nbf.v4.new_markdown_cell("## 2. Preprocessing\nConvert timestamps and handle basic data types.\n\n---"),
        nbf.v4.new_code_cell("df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])\ndf = df.sort_values(by=['CustomerID', 'InvoiceDate'])\n\n# We will use an observation window to predict future high-value conversion\ncutoff_date = df['InvoiceDate'].min() + pd.DateOffset(months=8)\nobs_window = df[df['InvoiceDate'] <= cutoff_date]\ntarget_window = df[df['InvoiceDate'] > cutoff_date]\nsnapshot_date = cutoff_date + dt.timedelta(days=1)"),
        nbf.v4.new_markdown_cell("## 3. Creating User Journey Features\nWe need to aggregate the touchpoints to the user level. Key features required:\n- Number of touchpoints/invoices per user\n- Recency (days since last purchase)\n- First Country (Channel analogue)\n- Unique products interacted with\n- Customer lifetime duration in observations\n\n---"),
        nbf.v4.new_code_cell("def aggregate_user_journey(user_df):\n    user_df = user_df.sort_values('InvoiceDate')\n    first_touch = user_df.iloc[0]\n    last_touch = user_df.iloc[-1]\n    \n    time_duration_days = (last_touch['InvoiceDate'] - first_touch['InvoiceDate']).days\n    \n    features = {\n        'CustomerID': first_touch['CustomerID'],\n        'num_invoices': user_df['InvoiceNo'].nunique(),\n        'total_quantity': user_df['Quantity'].sum(),\n        'recency_days': (snapshot_date - last_touch['InvoiceDate']).days,\n        'first_country': first_touch['Country'],\n        'unique_products': user_df['StockCode'].nunique(),\n        'historical_value': user_df['TotalPrice'].sum(),\n        'journey_duration_days': time_duration_days\n    }\n    return pd.Series(features)\n\nprint('Aggregating user journeys...')\nuser_features_df = obs_window.groupby('CustomerID').apply(aggregate_user_journey).reset_index(drop=True)\nprint('Done. User-level dataset shape:', user_features_df.shape)"),
        nbf.v4.new_markdown_cell("## 4. Encoding Categorical Variables\nWe will use one-hot encoding for categorical representations.\n\n---"),
        nbf.v4.new_code_cell("target_values = target_window.groupby('CustomerID')['TotalPrice'].sum()\n\n# Define 'converted' target as customers who spent more than $500 in the target period (High-Value Converters)\nuser_features_df['future_value'] = user_features_df['CustomerID'].map(target_values).fillna(0)\nuser_features_df['converted'] = (user_features_df['future_value'] > 500).astype(int)\n\n# Drop intermediate target mapped column\nuser_features_df.drop(columns=['future_value'], inplace=True)\n\nuser_features_encoded = pd.get_dummies(user_features_df, columns=['first_country'], drop_first=True)\ndisplay(user_features_encoded.head())"),
        nbf.v4.new_markdown_cell("## 5. Save Processed Dataset\n\n---"),
        nbf.v4.new_code_cell("output_path = '../data/processed_data.csv'\nuser_features_encoded.to_csv(output_path, index=False)\nprint(f'Processed dataset saved to {output_path}')")
    ]
    with open('../notebooks/feature_engineering.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

def create_modeling():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Customer Lifetime Value - Modeling & Explainability\n\nIn this notebook, we build classification models predicting future High-Value user conversion based on their user journey features. We will then explain the model using SHAP to infer feature attribution.\n\n---\n\n## 1. Import Libraries\n\n---"),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nfrom sklearn.model_selection import train_test_split, GridSearchCV\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import RandomForestClassifier\nfrom xgboost import XGBClassifier\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report\n\nimport shap\nimport pickle\nimport os\nimport warnings\nwarnings.filterwarnings('ignore')"),
        nbf.v4.new_markdown_cell("## 2. Load Processed Data & Train/Test Split\n\n---"),
        nbf.v4.new_code_cell("try:\n    df = pd.read_csv('../data/processed_data.csv')\nexcept Exception as e:\n    print('Error loading processed data:', e)\n\n# Features and Target\nX = df.drop(columns=['CustomerID', 'converted'])\ny = df['converted']\n\n# Train-Test Split\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n\nprint(f'Training set shape: {X_train.shape}')\nprint(f'Test set shape: {X_test.shape}')"),
        nbf.v4.new_markdown_cell("## 3. Baseline Models Training\n\n---"),
        nbf.v4.new_code_cell("models = {\n    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),\n    'Random Forest': RandomForestClassifier(random_state=42),\n    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)\n}\n\nresults = {}\n\nfor name, model in models.items():\n    model.fit(X_train, y_train)\n    y_pred = model.predict(X_test)\n    y_proba = model.predict_proba(X_test)[:, 1]\n    \n    acc = accuracy_score(y_test, y_pred)\n    prec = precision_score(y_test, y_pred)\n    rec = recall_score(y_test, y_pred)\n    f1 = f1_score(y_test, y_pred)\n    roc_auc = roc_auc_score(y_test, y_proba)\n    \n    results[name] = {\n        'Accuracy': acc,\n        'Precision': prec,\n        'Recall': rec,\n        'F1 Score': f1,\n        'ROC-AUC': roc_auc\n    }\n\nresults_df = pd.DataFrame(results).T\ndisplay(results_df)"),
        nbf.v4.new_markdown_cell("## 4. Hyperparameter Tuning (XGBoost)\nAssuming XGBoost performed best, we tune it using GridSearchCV.\n\n---"),
        nbf.v4.new_code_cell("param_grid = {\n    'n_estimators': [50, 100],\n    'max_depth': [3, 5],\n    'learning_rate': [0.01, 0.1]\n}\n\nxgb_tuned = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)\ngrid_search = GridSearchCV(estimator=xgb_tuned, param_grid=param_grid, scoring='roc_auc', cv=3, verbose=1)\n\nprint('Starting Grid Search...')\ngrid_search.fit(X_train, y_train)\n\nbest_model = grid_search.best_estimator_\nprint(f'Best Parameters: {grid_search.best_params_}')"),
        nbf.v4.new_markdown_cell("## 5. Model Evaluation\n\n---"),
        nbf.v4.new_code_cell("y_pred_best = best_model.predict(X_test)\ny_prob_best = best_model.predict_proba(X_test)[:, 1]\n\nprint('FINAL MODEL EVALUATION (XGBoost Tuned)')\nprint('Accuracy:', accuracy_score(y_test, y_pred_best))\nprint('Precision:', precision_score(y_test, y_pred_best))\nprint('Recall:', recall_score(y_test, y_pred_best))\nprint('F1 Score:', f1_score(y_test, y_pred_best))\nprint('ROC-AUC:', roc_auc_score(y_test, y_prob_best))\n\ncm = confusion_matrix(y_test, y_pred_best)\nplt.figure(figsize=(6, 4))\nsns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\nplt.title('Confusion Matrix')\nplt.xlabel('Predicted')\nplt.ylabel('Actual')\nplt.show()"),
        nbf.v4.new_markdown_cell("## 6. Saving Final Model\n\n---"),
        nbf.v4.new_code_cell("os.makedirs('../model', exist_ok=True)\nwith open('../model/trained_model.pkl', 'wb') as f:\n    pickle.dump(best_model, f)\nprint('Model saved to ../model/trained_model.pkl')"),
        nbf.v4.new_markdown_cell("## 7. Explainability: Feature Importance & SHAP\nThis helps us understand which features contribute the most to future conversions.\n\n---"),
        nbf.v4.new_code_cell("# Traditional Feature Importance\nfeature_importances = best_model.feature_importances_\nfeatures = X.columns\n\nimp_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False).head(10)\n\nplt.figure(figsize=(10, 6))\nsns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis')\nplt.title('Top 10 Feature Importance (XGBoost)')\nplt.show()"),
        nbf.v4.new_code_cell("# SHAP Values for Attribution\nexplainer = shap.TreeExplainer(best_model)\nshap_values = explainer.shap_values(X_test)\n\nshap.summary_plot(shap_values, X_test, feature_names=X.columns, plot_type='bar')\nplt.show()\n\nshap.summary_plot(shap_values, X_test, feature_names=X.columns)\nplt.show()")
    ]
    with open('../notebooks/modeling.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    os.makedirs('../notebooks', exist_ok=True)
    create_eda()
    create_feature_engineering()
    create_modeling()
    print("Notebooks strictly following the Sample Marketing Attribution full deep structure successfully generated.")
