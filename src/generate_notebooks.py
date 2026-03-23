import nbformat as nbf
import os

def create_eda():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Customer Lifetime Value - Exploratory Data Analysis (EDA)\n\nIn this notebook, we explore the `online_retail.csv` dataset to understand customer interactions and their impact on long-term value.\n\n---\n\n## 1. Import Libraries\n\n---"),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Set visualization style\nsns.set_theme(style='whitegrid')\nplt.rcParams['figure.figsize'] = (10, 6)\n\nimport warnings\nwarnings.filterwarnings('ignore')"),
        nbf.v4.new_markdown_cell("## 2. Load Dataset\n\n---"),
        nbf.v4.new_code_cell("try:\n    df = pd.read_csv('../data/online_retail.csv')\n    print('Dataset loaded successfully.')\n    display(df.head())\nexcept FileNotFoundError:\n    print('Error: ../data/online_retail.csv not found.')"),
        nbf.v4.new_markdown_cell("## 3. Dataset Overview and Missing Values\n\n---"),
        nbf.v4.new_code_cell("print('--- Dataset Info ---')\ndf.info()\n\nprint('\\n--- Missing Values ---')\ndisplay(df.isnull().sum())\n\nprint('\\n--- Descriptive Statistics ---')\ndisplay(df.describe(include='all'))\n\n# Quick cleaning for downstream EDA\ndf.dropna(subset=['CustomerID'], inplace=True)\ndf = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]"),
        nbf.v4.new_markdown_cell("## 4. Distribution of Marketing Channels (Countries)\n\n---"),
        nbf.v4.new_code_cell("plt.figure(figsize=(12, 5))\ntop_countries = df['Country'].value_counts().head(10)\nsns.barplot(x=top_countries.index, y=top_countries.values, palette='viridis')\nplt.title('Top 10 Countries by Transaction Count')\nplt.xlabel('Country')\nplt.ylabel('Count')\nplt.xticks(rotation=45)\nplt.show()"),
        nbf.v4.new_markdown_cell("## 5. Conversion Rate (High-Value Customers) per Country\n\n---"),
        nbf.v4.new_code_cell("df['TotalPrice'] = df['Quantity'] * df['UnitPrice']\ncountry_value = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)\n\nplt.figure(figsize=(10, 5))\nsns.barplot(x=country_value.index, y=country_value.values, palette='magma')\nplt.title('Total Revenue Generation per Country (Top 10)')\nplt.xlabel('Country')\nplt.ylabel('Revenue')\nplt.xticks(rotation=45)\nplt.show()"),
        nbf.v4.new_markdown_cell("## 6. Time-based Interaction Analysis\nLet's see how purchasing behaviors vary by time.\n\n---"),
        nbf.v4.new_code_cell("df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])\ndf['Hour'] = df['InvoiceDate'].dt.hour\ndf['DayOfWeek'] = df['InvoiceDate'].dt.day_name()\n\nplt.figure(figsize=(14, 5))\n\nplt.subplot(1, 2, 1)\nsns.countplot(data=df, x='Hour', color='steelblue')\nplt.title('Interactions by Hour of Day')\n\nplt.subplot(1, 2, 2)\ndays_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']\nsns.countplot(data=df, x='DayOfWeek', order=days_order, color='coral')\nplt.title('Interactions by Day of Week')\nplt.xticks(rotation=45)\n\nplt.tight_layout()\nplt.show()\n\ndf.to_csv('../data/online_retail_cleaned.csv', index=False)")
    ]
    with open('../notebooks/eda.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

def create_feature_engineering():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Customer Lifetime Value - Feature Engineering\n\nIn this notebook, we transform the raw transactional data into user-level features to prepare for modeling.\n\n---\n\n## 1. Import Libraries and Load Data\n\n---"),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport datetime as dt\n\ntry:\n    df = pd.read_csv('../data/online_retail_cleaned.csv')\n    print('Data loaded.')\nexcept Exception as e:\n    print('Failed to load data:', e)"),
        nbf.v4.new_markdown_cell("## 2. Preprocessing\nConvert timestamps and handle basic data types.\n\n---"),
        nbf.v4.new_code_cell("df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])\ndf = df.sort_values(by=['CustomerID', 'InvoiceDate'])\n\ncutoff_date = df['InvoiceDate'].min() + pd.DateOffset(months=8)\nobs_window = df[df['InvoiceDate'] <= cutoff_date]\ntarget_window = df[df['InvoiceDate'] > cutoff_date]\nsnapshot_date = cutoff_date + dt.timedelta(days=1)"),
        nbf.v4.new_markdown_cell("## 3. Creating User Journey Features\n\n---"),
        nbf.v4.new_code_cell("def aggregate_user_journey(user_df):\n    user_df = user_df.sort_values('InvoiceDate')\n    first_touch = user_df.iloc[0]\n    last_touch = user_df.iloc[-1]\n    time_duration_days = (last_touch['InvoiceDate'] - first_touch['InvoiceDate']).days\n    features = {\n        'CustomerID': first_touch['CustomerID'],\n        'num_invoices': user_df['InvoiceNo'].nunique(),\n        'total_quantity': user_df['Quantity'].sum(),\n        'recency_days': (snapshot_date - last_touch['InvoiceDate']).days,\n        'first_country': first_touch['Country'],\n        'unique_products': user_df['StockCode'].nunique(),\n        'historical_value': user_df['TotalPrice'].sum(),\n        'journey_duration_days': time_duration_days\n    }\n    return pd.Series(features)\n\nprint('Aggregating user journeys...')\nuser_features_df = obs_window.groupby('CustomerID').apply(aggregate_user_journey).reset_index(drop=True)\nprint('Done. User-level dataset shape:', user_features_df.shape)"),
        nbf.v4.new_markdown_cell("## 4. Encoding Categorical Variables\n\n---"),
        nbf.v4.new_code_cell("target_values = target_window.groupby('CustomerID')['TotalPrice'].sum()\nuser_features_df['future_value'] = user_features_df['CustomerID'].map(target_values).fillna(0)\nuser_features_df['converted'] = (user_features_df['future_value'] > 500).astype(int)\nuser_features_df.drop(columns=['future_value'], inplace=True)\n\nuser_features_encoded = pd.get_dummies(user_features_df, columns=['first_country'], drop_first=True)\ndisplay(user_features_encoded.head())"),
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
        nbf.v4.new_code_cell("try:\n    df = pd.read_csv('../data/processed_data.csv')\nexcept Exception as e:\n    print('Error loading processed data:', e)\n\nX = df.drop(columns=['CustomerID', 'converted'])\ny = df['converted']\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n\nprint(f'Training set shape: {X_train.shape}')\nprint(f'Test set shape: {X_test.shape}')"),
        nbf.v4.new_markdown_cell("## 3. Baseline Models Training\n\n---"),
        nbf.v4.new_code_cell("models = {\n    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),\n    'Random Forest': RandomForestClassifier(random_state=42),\n    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)\n}\n\nresults = {}\n\n# Adding Model Checkpoints (Saving each baseline model trained state)\nos.makedirs('../model/checkpoints', exist_ok=True)\n\nfor name, model in models.items():\n    model.fit(X_train, y_train)\n    \n    # Explicitly Save Checkpoints\n    safe_name = name.replace(' ', '_').lower()\n    checkpoint_path = f'../model/checkpoints/{safe_name}_checkpoint.pkl'\n    with open(checkpoint_path, 'wb') as f:\n        pickle.dump(model, f)\n    print(f'Saved {name} checkpoint to {checkpoint_path}')\n\n    y_pred = model.predict(X_test)\n    y_proba = model.predict_proba(X_test)[:, 1]\n    \n    acc = accuracy_score(y_test, y_pred)\n    prec = precision_score(y_test, y_pred)\n    rec = recall_score(y_test, y_pred)\n    f1 = f1_score(y_test, y_pred)\n    roc_auc = roc_auc_score(y_test, y_proba)\n    \n    results[name] = {\n        'Accuracy': acc,\n        'Precision': prec,\n        'Recall': rec,\n        'F1 Score': f1,\n        'ROC-AUC': roc_auc\n    }\n\nresults_df = pd.DataFrame(results).T\ndisplay(results_df)"),
        nbf.v4.new_markdown_cell("## 4. Hyperparameter Tuning (XGBoost)\n\n---"),
        nbf.v4.new_code_cell("param_grid = {\n    'n_estimators': [50, 100],\n    'max_depth': [3, 5],\n    'learning_rate': [0.01, 0.1]\n}\n\nxgb_tuned = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)\ngrid_search = GridSearchCV(estimator=xgb_tuned, param_grid=param_grid, scoring='roc_auc', cv=3, verbose=1)\n\nprint('Starting Grid Search...')\ngrid_search.fit(X_train, y_train)\nbest_model = grid_search.best_estimator_\nprint(f'Best Parameters: {grid_search.best_params_}')"),
        nbf.v4.new_markdown_cell("## 5. Model Evaluation\n\n---"),
        nbf.v4.new_code_cell("y_pred_best = best_model.predict(X_test)\ny_prob_best = best_model.predict_proba(X_test)[:, 1]\n\nprint('FINAL MODEL EVALUATION (XGBoost Tuned)')\nprint('Accuracy:', accuracy_score(y_test, y_pred_best))\nprint('Precision:', precision_score(y_test, y_pred_best))\nprint('Recall:', recall_score(y_test, y_pred_best))\nprint('F1 Score:', f1_score(y_test, y_pred_best))\nprint('ROC-AUC:', roc_auc_score(y_test, y_prob_best))\n\ncm = confusion_matrix(y_test, y_pred_best)\nplt.figure(figsize=(6, 4))\nsns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\nplt.title('Confusion Matrix')\nplt.xlabel('Predicted')\nplt.ylabel('Actual')\nplt.show()"),
        nbf.v4.new_markdown_cell("## 6. Saving Final Model\n\n---"),
        nbf.v4.new_code_cell("with open('../model/trained_model.pkl', 'wb') as f:\n    pickle.dump(best_model, f)\nprint('Model saved to ../model/trained_model.pkl')"),
        nbf.v4.new_markdown_cell("## 7. Explainability: Feature Importance & SHAP\n\n---"),
        nbf.v4.new_code_cell("# Traditional Feature Importance\nfeature_importances = best_model.feature_importances_\nfeatures = X.columns\nimp_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False).head(10)\nplt.figure(figsize=(10, 6))\nsns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis')\nplt.title('Top 10 Feature Importance (XGBoost)')\nplt.show()"),
        nbf.v4.new_code_cell("# SHAP Values for Attribution\nexplainer = shap.TreeExplainer(best_model)\nshap_values = explainer.shap_values(X_test)\nshap.summary_plot(shap_values, X_test, feature_names=X.columns, plot_type='bar')\nplt.show()\nshap.summary_plot(shap_values, X_test, feature_names=X.columns)\nplt.show()")
    ]
    with open('../notebooks/modeling.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    os.makedirs('../notebooks', exist_ok=True)
    create_eda()
    create_feature_engineering()
    create_modeling()
    print("Notebooks strictly following the Sample Marketing Attribution full deep structure successfully generated.")
