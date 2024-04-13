import pandas as pd
df = pd.read_csv(r"C:\Users\MERT\Downloads\diabetes_prediction_dataset.csv")


categorical_columns = ['gender', 'hypertension', 'heart_disease', 'smoking_history', 'diabetes']
for column in categorical_columns:
    mode_value = df[column].mode()[0]
    print(f"The mode for {column} is {mode_value}")
