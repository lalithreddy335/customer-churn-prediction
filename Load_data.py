import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# Load CSV
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Encode password properly
password = quote_plus('Lalith@08')

# Connect to MySQL
engine = create_engine(f'mysql+pymysql://root:{password}@localhost/churn_db')

# Load into MySQL
df.to_sql('customers', con=engine, if_exists='replace', index=False)

print("Data loaded successfully!")