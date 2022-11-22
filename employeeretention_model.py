# Importing the required packages
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Importing the data
ipc = pd.read_csv(r'HR-Employee-Attrition.csv',encoding="utf8")
ipc1 = ipc.drop(['YearsSinceLastPromotion', 'PerformanceRating', 'MonthlyIncome', 'Education', 'Gender'
                 ,'JobSatisfaction', 'DailyRate', 'CostOfHiring','TotalWorkingYears'], axis = 1)

# Ordinal feature encoding
df = ipc1.copy()
target = 'Attrition'
encode = ['BusinessTravel','Department','EducationField','EnvironmentSatisfaction','OverTime','JobRole',
          'MaritalStatus']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

# Target mapping
target_mapper = {'No':0, 'Yes':1}
def target_encode(val):
    return target_mapper[val]

df['Attrition'] = df['Attrition'].apply(target_encode)

# Separating X and y
X = df.drop('Attrition', axis=1)
Y = df['Attrition']

# Build random forest model
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
pickle.dump(clf, open('ipc_clf.pkl', 'wb'))
