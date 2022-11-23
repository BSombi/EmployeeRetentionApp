from fastapi import FastAPI
import uvicorn
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel
  
# Creating FastAPI instance
app = FastAPI()
  
# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    Final_branch : str
    Highest_Sales : float
    Lowest_Sales : float
    Average_Sales : float
    Gender : str
    Marital_Status : str
    BirthMonth : int
    Age : int
    House : str
    Loan_Type : str
    Fund : str
    Loan_Purpose : str
    Client_Type : str
    Client_Classification : str
    Currency : str
    principal_amount : float
    Expected_No_Repayments : int
    Interest_Calculated_in_Period : str
    Repayment_Frequency_Period : str
    Month_Borrowed : int
    Quarter_Borrowed : int
    Days_to_RePurchase : float
  
# Loading Moneymart Dataset
moneymart_first = pd.read_csv(r'moneymart_cleaned.csv',encoding="utf8")

df1 = moneymart_first.copy()
target = 'target'

df = pd.get_dummies(df1, columns = ['Final_branch','Gender','Marital_Status','House','Loan_Type','Fund','Loan_Purpose','Client_Type',
          'Client_Classification','Currency','Interest_Calculated_in_Period','Repayment_Frequency_Period'])

target_mapper = {'Default':0, 'Non-Default':1}
def target_encode(val):
    return target_mapper[val]

df['target'] = df['target'].apply(target_encode)

# Replace the N/a class with class 'missing'
df['Highest_Sales'] = np.where(df['Highest_Sales'].isnull(), '0', df['Highest_Sales'])
df['Lowest_Sales'] = np.where(df['Lowest_Sales'].isnull(), '0', df['Lowest_Sales'])
df['Average_Sales'] = np.where(df['Average_Sales'].isnull(), '0', df['Average_Sales'])
df['BirthMonth'] = np.where(df['BirthMonth'].isnull(), '0', df['BirthMonth'])
df['Age'] = np.where(df['Age'].isnull(), '0', df['Age'])
df['Expected_No_Repayments'] = np.where(df['Expected_No_Repayments'].isnull(), '0', df['Expected_No_Repayments'])
df['Month_Borrowed'] = np.where(df['Month_Borrowed'].isnull(), '0', df['Month_Borrowed'])
df['Quarter_Borrowed'] = np.where(df['Quarter_Borrowed'].isnull(), '0', df['Quarter_Borrowed'])
df['Days_to_RePurchase'] = np.where(df['Days_to_RePurchase'].isnull(), '0', df['Days_to_RePurchase'])

 
# Separating X and y
X = df.drop('target', axis=1)
Y = df['target']
  
# Creating and Fitting our Model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)
  
# Creating an Endpoint to recieve the data
# to make prediction on.
@app.post('/predict')
def predict(data : request_body):
    # Making the data in a form suitable for prediction
    test_data = [[
            data.Final_branch,
            data.Highest_Sales,
            data.Lowest_Sales,
            data.Average_Sales,
            data.Gender,
            data.Marital_Status,
            data.BirthMonth,
            data.Age,
            data.House,
            data.Loan_Type,
            data.Fund,
            data.Loan_Purpose,
            data.Client_Type,
            data.Client_Classification,
            data.Currency,
            data.principal_amount,
            data.Expected_No_Repayments,
            data.Interest_Calculated_in_Period,
            data.Repayment_Frequency_Period,
            data.Month_Borrowed,
            data.Quarter_Borrowed,
            data.Days_to_RePurchase,
    ]]
    features = pd.DataFrame(test_data, index=[0])
    #return features
        
    # Combines user input features with entire loans dataset
    # This will be useful for the encoding phase
    moneymart_raw = pd.read_csv(r'C:\Users\Benjamin\Documents\Benjamin\Benjamin Sombi Docs\D\Dreatol Fintech Material\StreamLit App\Moneymart App\moneymart_cleaned.csv')
    moneymart = moneymart_raw.drop(columns=['target'])
    df2 = pd.concat([features,moneymart],axis=0)

   # Encoding of ordinal features
    encode = ['Final_branch','Gender','Marital_Status','House','Loan_Type','Fund','Loan_Purpose','Client_Type',
          'Client_Classification','Currency','Interest_Calculated_in_Period','Repayment_Frequency_Period']
    for col in encode:
      dummy = pd.get_dummies(df2[col], prefix=col)
      df2 = pd.concat([df2,dummy], axis=1)
      del df2[col]
    df2 = df2[:1] # Selects only the first row (the user input data)
           
      
    # Predicting the Class
    class_idx = clf.predict(df2)[0]
      
    # Return the Result
    return { 'target' : moneymart_raw.target_names[class_idx]}
