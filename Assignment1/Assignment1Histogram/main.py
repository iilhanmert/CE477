import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

data = pd.read_csv("/Users/canbaytekin/Documents/GitHub/CE477/data/diabetes_prediction_dataset.csv")

plt.hist(data['gender'])
plt.title("Gender Histogram")
plt.xlabel("Gender")
plt.ylabel("Frequency")

plt.hist(data['smoking_history'])
plt.title("Smoking History")
plt.xlabel("Smoking History")
plt.ylabel("Frequency")

# 2nd way of displaying the graph
#plt.rcParams["figure.autolayout"] = True
#plt.show()

fig1 = px.histogram(data, x="gender")
fig1.show()

fig2 = px.histogram(data, x="smoking_history")
fig2.show()
