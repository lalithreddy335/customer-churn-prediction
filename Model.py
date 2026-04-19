import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
import pickle

# Connect to MySQL
password = quote_plus('Lalith@08')
engine = create_engine(f'mysql+pymysql://root:{password}@localhost/churn_db')
df = pd.read_sql('SELECT * FROM customers', engine)

# Data Preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Features and Target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1 — Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_pred):.2f}")

# Model 2 — Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.2f}")

# Model 3 — XGBoost
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_pred):.2f}")
print(f"XGBoost AUC-ROC: {roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1]):.2f}")
print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_pred))

# Save best model
with open('models/model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

print("\nModel saved successfully!")