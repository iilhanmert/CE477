import pandas as pd

data = pd.read_csv("/Users/canbaytekin/Documents/GitHub/CE477/data/diabetes_prediction_dataset.csv")

print(data.isna())
print(data.isnull())
print(data == float('nan'))
print(data == str('No Info'))
print(data.isnull().any())

