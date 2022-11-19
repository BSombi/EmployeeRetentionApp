# 1. Library imports
import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# 2. Create the app object
app = FastAPI()

moneymart = pd.read_csv(r'C:\Users\Benjamin\Documents\Benjamin\Benjamin Sombi Docs\D\Dreatol Fintech Material\StreamLit App\Moneymart App\moneymart_cleaned.csv',encoding="utf8")
#. Load trained Pipeline
model = load_model('moneymart-pipeline')

# Define predict function
@app.post('/predict')
def predict(Final_branch, Highest_Sales, Lowest_Sales, Average_Sales, Gender, Marital_Status, BirthMonth, Age, House, Loan_Type, Fund, Loan_Purpose, Client_Type, Client_Classification, Currency, principal_amount, Expected_No_Repayments, Interest_Calculated_in_Period, Repayment_Frequency_Period, Month_Borrowed, Quarter_Borrowed, Days_to_RePurchase):
    data = pd.DataFrame([[Final_branch, Highest_Sales, Lowest_Sales, Average_Sales, Gender, Marital_Status, BirthMonth, Age, House, Loan_Type, Fund, Loan_Purpose, Client_Type, Client_Classification, Currency, principal_amount, Expected_No_Repayments, Interest_Calculated_in_Period, Repayment_Frequency_Period, Month_Borrowed, Quarter_Borrowed, Days_to_RePurchase]])
    data.columns = ['Final_branch', 'Highest_Sales', 'Lowest_Sales', 'Average_Sales', 'Gender', 'Marital_Status', 'BirthMonth', 'Age', 'House', 'Loan_Type', 'Fund', 'Loan_Purpose', 'Client_Type', 'Client_Classification', 'Currency', 'principal_amount', 'Expected_No_Repayments', 'Interest_Calculated_in_Period', 'Repayment_Frequency_Period', 'Month_Borrowed', 'Quarter_Borrowed', 'Days_to_RePurchase']

    predictions = predict_model(model, data=data)
    return {'prediction': (predictions['Label'][0])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)