import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/canbaytekin/Documents/GitHub/CE477/data/diabetes_prediction_dataset.csv")

columns = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']

plt.boxplot(data[columns].values)
plt.xticks(range(1, len(columns) + 1), columns, rotation=45)
plt.title("Boxplots")

plt.show()
