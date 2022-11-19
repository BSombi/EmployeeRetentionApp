from nbformat import write
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport



st.write("""
# Data Profiling App
**This app Explore the Data before modeling**
""")

st.sidebar.header('User Input Features')


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    
    profile = ProfileReport(input_df,

                       title="New Data for profiling",

    )

    st.title("Detailed Report of the Data Used")

    st.write(input_df)

    st_profile_report(profile)    
    
else:
    st.write("You did not upload the new file")
#    input_df = pd.read_csv(r'C:\Users\Benjamin\Documents\Benjamin\Benjamin Sombi Docs\IPC Staff\Cluster Documents\StreamLit App\Employee Retention App\HR-Employee-Attrition.csv',encoding="utf8")

