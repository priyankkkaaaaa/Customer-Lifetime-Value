import nbformat as nbf
import os

def create_eda():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Customer Lifetime Value - Exploratory Data Analysis (EDA)\nIn this notebook, we explore the online_retail dataset to identify core trends, perform descriptive statistics, and prepare the dataset for feature engineering."),
        nbf.v4.new_markdown_cell("## 1. Import Libraries"),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\nsns.set_theme(style='whitegrid')"),
        nbf.v4.new_markdown_cell("## 2. Load Dataset"),
        nbf.v4.new_code_cell("try:\n    df = pd.read_csv('../data/online_retail.csv')\n    print('Dataset loaded successfully.')\n    display(df.head())\nexcept FileNotFoundError:\n    print('Error loading dataset.')"),
        nbf.v4.new_markdown_cell("## 3. Dataset Overview and Missing Values"),
        nbf.v4.new_code_cell("print('\\n--- Missing Values ---')\ndisplay(df.isnull().sum())\nprint('\\n--- Descriptive Statistics ---')\ndisplay(df.describe())\n\ndf.dropna(subset=['CustomerID'], inplace=True)\ndf = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]\nprint('\\nCleaned dataset shape:', df.shape)"),
        nbf.v4.new_markdown_cell("## 4. Distribution of Top Countries"),
        nbf.v4.new_code_cell("plt.figure(figsize=(10, 5))\ntop_countries = df['Country'].value_counts().head(10)\nsns.barplot(x=top_countries.index, y=top_countries.values, palette='viridis')\nplt.title('Top 10 Countries by Transaction Count')\nplt.xticks(rotation=45)\nplt.show()"),
        nbf.v4.new_markdown_cell("## 5. Monetary Value Distribution"),
        nbf.v4.new_code_cell("df['TotalPrice'] = df['Quantity'] * df['UnitPrice']\ncountry_value = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)\nplt.figure(figsize=(10, 5))\nsns.barplot(x=country_value.index, y=country_value.values, palette='magma')\nplt.title('Total Revenue Generation per Country (Top 10)')\nplt.xticks(rotation=45)\nplt.show()"),
        nbf.v4.new_markdown_cell("## 6. Time-based Interaction Analysis"),
        nbf.v4.new_code_cell("df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])\ndf['Hour'] = df['InvoiceDate'].dt.hour\ndf['DayOfWeek'] = df['InvoiceDate'].dt.day_name()\n\nplt.figure(figsize=(12, 4))\nplt.subplot(1, 2, 1)\nsns.countplot(data=df, x='Hour', color='steelblue')\nplt.title('Interactions by Hour of Day')\n\nplt.subplot(1, 2, 2)\ndays_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']\nsns.countplot(data=df, x='DayOfWeek', order=days_order, color='coral')\nplt.title('Interactions by Day of Week')\nplt.xticks(rotation=45)\nplt.show()\n\ndf.to_csv('../data/online_retail_cleaned.csv', index=False)")
    ]
    with open('../notebooks/eda.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

def create_feature_engineering():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Customer Lifetime Value - Feature Engineering\nIn this notebook, we transform the raw transactional data into Customer-level features (RFM) and encode targets."),
        nbf.v4.new_markdown_cell("## 1. Import Libraries and Load Data"),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport datetime as dt\n\ntry:\n    df = pd.read_csv('../data/online_retail_cleaned.csv')\n    print('Cleaned data loaded.')\nexcept:\n    print('Data failed to load.')"),
        nbf.v4.new_markdown_cell("## 2. Preprocessing & Time Splitting"),
        nbf.v4.new_code_cell("df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])\ncutoff_date = df['InvoiceDate'].min() + pd.DateOffset(months=8)\nobs_window = df[df['InvoiceDate'] <= cutoff_date]\ntarget_window = df[df['InvoiceDate'] > cutoff_date]\nsnapshot_date = cutoff_date + dt.timedelta(days=1)"),
        nbf.v4.new_markdown_cell("## 3. Creating User Journey Features (RFM)"),
        nbf.v4.new_code_cell("def aggregate_user_journey(user_df):\n    user_df = user_df.sort_values('InvoiceDate')\n    first_touch = user_df.iloc[0]\n    last_touch = user_df.iloc[-1]\n    features = {\n        'CustomerID': first_touch['CustomerID'],\n        'num_invoices': user_df['InvoiceNo'].nunique(),\n        'total_quantity': user_df['Quantity'].sum(),\n        'recency_days': (snapshot_date - last_touch['InvoiceDate']).days,\n        'first_country': first_touch['Country'],\n        'historical_value': user_df['TotalPrice'].sum(),\n        'time_to_conversion_days': (last_touch['InvoiceDate'] - first_touch['InvoiceDate']).days\n    }\n    return pd.Series(features)\n\nprint('Aggregating user journeys...')\nuser_features_df = obs_window.groupby('CustomerID').apply(aggregate_user_journey).reset_index(drop=True)\nprint('Done. User-level dataset shape:', user_features_df.shape)"),
        nbf.v4.new_markdown_cell("## 4. Encoding Targets & Final Export"),
        nbf.v4.new_code_cell("target_values = target_window.groupby('CustomerID')['TotalPrice'].sum()\nuser_features_df['future_value'] = user_features_df['CustomerID'].map(target_values).fillna(0)\n\n# Binary target mapping to 'High Value' -> mapping closely to conversion standard\nuser_features_df['converted'] = (user_features_df['future_value'] > 500).astype(int)\nuser_features_df.drop(columns=['future_value'], inplace=True)\n\nuser_features_encoded = pd.get_dummies(user_features_df, columns=['first_country'], drop_first=True)\nuser_features_encoded.to_csv('../data/processed_data.csv', index=False)\nprint('Processed dataset mapped and saved.')\ndisplay(user_features_encoded.head())")
    ]
    with open('../notebooks/feature_engineering.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

def create_modeling():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Customer Lifetime Value - Modeling & Explainability\nIn this notebook, we classify High-Value user conversions based on the engineered tracking dataset, evaluate them comprehensively, and explain via SHAP."),
        nbf.v4.new_markdown_cell("## 1. Import Libraries"),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.model_selection import train_test_split, GridSearchCV\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import RandomForestClassifier\nfrom xgboost import XGBClassifier\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix\nimport shap\nimport pickle\nimport os\nimport warnings\nwarnings.filterwarnings('ignore')\n\nsns.set_theme(style='whitegrid')"),
        nbf.v4.new_markdown_cell("## 2. Load Processed Data & Train/Test Split"),
        nbf.v4.new_code_cell("df = pd.read_csv('../data/processed_data.csv')\n\nX = df.drop(columns=['CustomerID', 'converted'])\ny = df['converted']\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\nprint(f'Training shape: {X_train.shape}, Testing shape: {X_test.shape}')"),
        nbf.v4.new_markdown_cell("## 3. Baseline Models Training\nEvaluate classical regressions against tree ensembles."),
        nbf.v4.new_code_cell("models = {\n    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),\n    'Random Forest': RandomForestClassifier(random_state=42),\n    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)\n}\n\nresults = {}\nos.makedirs('../model/checkpoints', exist_ok=True)\n\nfor name, model in models.items():\n    model.fit(X_train, y_train)\n    \n    # Checkpointing model training progression immediately\n    with open(f'../model/checkpoints/{name.replace(\" \", \"_\").lower()}_checkpoint.pkl', 'wb') as f:\n        pickle.dump(model, f)\n        \n    y_pred = model.predict(X_test)\n    y_proba = model.predict_proba(X_test)[:, 1]\n    \n    results[name] = {\n        'Accuracy': accuracy_score(y_test, y_pred),\n        'Precision': precision_score(y_test, y_pred),\n        'Recall': recall_score(y_test, y_pred),\n        'F1 Score': f1_score(y_test, y_pred),\n        'ROC-AUC': roc_auc_score(y_test, y_proba)\n    }\n\nresults_df = pd.DataFrame(results).T\ndisplay(results_df)"),
        nbf.v4.new_markdown_cell("## 4. Hyperparameter Tuning (XGBoost)\nGridSearch optimizing for High-Value ROC_AUC."),
        nbf.v4.new_code_cell("param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}\n\nxgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)\ngrid = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='roc_auc', cv=3, verbose=1)\ngrid.fit(X_train, y_train)\n\nbest_model = grid.best_estimator_\nprint('Best Parameters:', grid.best_params_)"),
        nbf.v4.new_markdown_cell("## 5. Model Evaluation (Tuned)\nValidation via strict classification matrices."),
        nbf.v4.new_code_cell("y_pred_best = best_model.predict(X_test)\ny_prob_best = best_model.predict_proba(X_test)[:, 1]\n\nprint('--- XGBoost Tuned Evaluation ---')\nprint(f'Accuracy:  {accuracy_score(y_test, y_pred_best):.4f}')\nprint(f'Precision: {precision_score(y_test, y_pred_best):.4f}')\nprint(f'Recall:    {recall_score(y_test, y_pred_best):.4f}')\nprint(f'F1 Score:  {f1_score(y_test, y_pred_best):.4f}')\nprint(f'ROC-AUC:   {roc_auc_score(y_test, y_prob_best):.4f}')\n\ncm = confusion_matrix(y_test, y_pred_best)\nplt.figure(figsize=(5, 3))\nsns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\nplt.title('Confusion Matrix')\nplt.ylabel('Actual')\nplt.xlabel('Predicted')\nplt.show()"),
        nbf.v4.new_markdown_cell("## 6. Saving Final Model\nSerialize to disk immediately for application deployment."),
        nbf.v4.new_code_cell("with open('../model/trained_model.pkl', 'wb') as f:\n    pickle.dump(best_model, f)\nprint('Model securely saved.')"),
        nbf.v4.new_markdown_cell("## 7. Explainability: Feature Importance & SHAP\nInterpret the top drivers of Customer Value."),
        nbf.v4.new_code_cell("importances = pd.DataFrame({'Feature': X.columns, 'Importance': best_model.feature_importances_}).sort_values('Importance', ascending=False).head(10)\n\nplt.figure(figsize=(10, 5))\nsns.barplot(x='Importance', y='Feature', data=importances, palette='viridis')\nplt.title('Top 10 Traditional Feature Importances')\nplt.show()"),
        nbf.v4.new_code_cell("explainer = shap.TreeExplainer(best_model)\nshap_values = explainer.shap_values(X_test)\n\nshap.summary_plot(shap_values, X_test, feature_names=X.columns, plot_type='bar')\nshap.summary_plot(shap_values, X_test, feature_names=X.columns)")
    ]
    with open('../notebooks/modeling.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    create_eda()
    create_feature_engineering()
    create_modeling()
    print('Directly generated all final notebooks strictly without Day labels and with robust logic.')
