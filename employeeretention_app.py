import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import *


st.write("""
# Employee Attrition Prediction App

**This app predicts the chances of employee leaving your organisation.**
""")

st.header('User Input Features')

# Collects user input features into dataframe
def user_input_features():
    Age = st.slider('Age', 18,80,30)
    BusinessTravel = st.selectbox(' Business Travel',('Non-Travel','Travel_Frequently','Travel_Rarely'))
    Department = st.selectbox('Department',('Human Resources','Research & Development','Sales'))
    EducationField = st.selectbox('Education Field',('Human Resources','Life Sciences','Marketing','Medical','Technical Degree','Other'))
    EnvironmentSatisfaction = st.selectbox('Environment Satisfaction',('Low','Medium','High','Very High'))
    JobRole = st.selectbox('Job Role',('Healthcare Representative','Human Resources','Laboratory Technician','Manager','Manufacturing Director','Research Director','Research Scientist','Sales Executive','Sales Representative'))
    MaritalStatus = st.selectbox('Marital Status',('Married','Single','Divorced'))
    NumCompaniesWorked = st.slider('Number Of Companies Worked', 0,40,4)
    OverTime = st.selectbox('Over Time',('Yes','No'))
    TrainingTimesLastYear = st.slider('Training Times Last Year', 0,10,2)
    YearsAtCompany = st.slider('Years At Company', 0,30,2)
    YearsWithCurrManager = st.slider('Years With Current Manager', 0,30,2)
        
       
    data = {'Age': Age,
            'BusinessTravel': BusinessTravel,
            'Department': Department,
            'EducationField': EducationField,
            'EnvironmentSatisfaction': EnvironmentSatisfaction,
            'JobRole': JobRole,
            'MaritalStatus': MaritalStatus,
            'NumCompaniesWorked': NumCompaniesWorked,
            'OverTime': OverTime,
            'TrainingTimesLastYear': TrainingTimesLastYear,
            'YearsAtCompany': YearsAtCompany,
            'YearsWithCurrManager': YearsWithCurrManager}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
ipc_raw = pd.read_csv(r'HR-Employee-Attrition.csv',encoding="utf8")
ipc = ipc_raw.drop(columns=['Attrition','YearsSinceLastPromotion', 'PerformanceRating', 'MonthlyIncome', 'Education', 
                            'Gender', 'JobSatisfaction', 'DailyRate', 'CostOfHiring','TotalWorkingYears'])
df = pd.concat([input_df,ipc],axis=0)

# Encoding of ordinal features
encode = ['BusinessTravel','Department','EducationField','EnvironmentSatisfaction','OverTime','JobRole',
          'MaritalStatus']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)


# Reads in saved classification model
load_clf = pickle.load(open('ipc_clf.pkl', 'rb'))


prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df).max()
    
col1, col2 = st.columns(2)
    
# Display Prediction Results
with col1:
    st.subheader('Prediction')
    if prediction == 0:
       st.write('The Prediction is **No**. Therefore the employee will not leave the organisation')
    else:
       st.write('The Prediction is **Yes**. Therefore the employee will leave the organisation')
    
with col2:
    st.subheader('Prediction Probability')
    if prediction == 0:
       st.write('The Probability of the employee staying within the organisation is', "{:.0%}".format(prediction_proba))
    else:
       st.write('The Probability of the employee leaving the organisation is',"{:.0%}".format(prediction_proba))
