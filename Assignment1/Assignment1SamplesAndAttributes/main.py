import pandas as pd

df = pd.read_csv(r"C:\Users\MERT\Downloads\diabetes_prediction_dataset.csv")
print(df.head())

num_samples, num_attributes = df.shape
print(f"The dataset contains {num_samples} samples and {num_attributes} attributes.")
