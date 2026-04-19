# Customer Churn Prediction

End-to-end machine learning project to predict customer churn using Python, MySQL, XGBoost, and Streamlit.

## Problem Statement
Telecom companies lose significant revenue due to customer churn. This project analyzes 7,000+ customer records to identify churn drivers and predict which customers are likely to leave.

## Key Insights
- Overall churn rate: 26.5%
- Fiber optic customers churn at 41.89% vs DSL at 18.96%
- Customers who churn have median tenure of 10 months vs 38 months for loyal customers
- Month-to-month contracts have significantly higher churn rates

## Tech Stack
- **Python** — Pandas, NumPy, Scikit-learn, XGBoost, Seaborn
- **MySQL** — Data storage and SQL analysis
- **Streamlit** — Interactive web app deployment

## Project Structure
- `Load_data.py` — Load CSV into MySQL
- `eda.py` — Exploratory Data Analysis
- `Model.py` — ML model training
- `app.py` — Streamlit web app

## Model Performance
| Model | Accuracy | AUC-ROC |
|---|---|---|
| Logistic Regression | 0.80 | - |
| Random Forest | 0.79 | - |
| XGBoost | 0.76 | 0.81 |

## Live Demo
Add Streamlit Cloud link here

## How to Run
pip install -r requirements.txt
streamlit run app.py