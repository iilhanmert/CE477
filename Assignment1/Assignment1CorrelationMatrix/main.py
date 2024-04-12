import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/canbaytekin/Documents/GitHub/CE477/data/diabetes_prediction_dataset.csv")

matrix = data.corr(numeric_only=True)
sn.heatmap(matrix, annot=True)
plt.show()