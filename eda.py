import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# Connect to MySQL
password = quote_plus('Lalith@08')
engine = create_engine(f'mysql+pymysql://root:{password}@localhost/churn_db')

# Load data
df = pd.read_sql('SELECT * FROM customers', engine)

# Fix TotalCharges column
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Plot 1 — Churn Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df, palette='Set2')
plt.title('Churn Distribution')
plt.savefig('churn_distribution.png')
plt.show()

# Plot 2 — Churn by Internet Service
plt.figure(figsize=(8,5))
sns.countplot(x='InternetService', hue='Churn', data=df, palette='Set2')
plt.title('Churn by Internet Service')
plt.savefig('churn_by_internet.png')
plt.show()

# Plot 3 — Tenure vs Churn
plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='tenure', data=df, palette='Set2')
plt.title('Tenure vs Churn')
plt.savefig('tenure_vs_churn.png')
plt.show()

print("EDA completed!")