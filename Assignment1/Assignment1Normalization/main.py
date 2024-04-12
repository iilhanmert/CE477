from sklearn import preprocessing as preproc
import pandas as pd
import numpy as np

data = pd.read_csv("/Users/canbaytekin/Documents/GitHub/CE477/data/diabetes_prediction_dataset.csv")

newData = data.select_dtypes(include=np.number)
dataColumn = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']

scaler = preproc.MinMaxScaler()
normalize = scaler.fit_transform(newData)
normalized_df = pd.DataFrame(normalize, columns=dataColumn)

print('Original Data : \n', newData)
print('Normalized Data : \n', normalized_df)

scaler = preproc.StandardScaler()
standardize = scaler.fit_transform(newData)
standardized_df = pd.DataFrame(standardize, columns=dataColumn)

print('Standardized Data : \n', standardized_df)

