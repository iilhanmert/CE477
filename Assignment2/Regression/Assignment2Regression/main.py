import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn import preprocessing as preproc

data = pd.read_csv("/Users/canbaytekin/Documents/GitHub/CE477/data/diabetes_prediction_dataset.csv")

df = pd.DataFrame(data)

dataColumn = ['bmi', 'blood_glucose_level']

X = df[["bmi", "blood_glucose_level"]].values
y = df['HbA1c_level'].values

scaler = preproc.MinMaxScaler()
normalize = scaler.fit_transform(X)
normalized_df = pd.DataFrame(normalize, columns=dataColumn)
print(normalized_df)

pca = PCA(n_components=1)
X = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
predicted = regr.predict(X_test)

print()
print("Coefficients: ", regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, predicted))
print("Coefficient of determination: %.2f" % r2_score(y_test, predicted))

plt.scatter(X_test, y_test, color="black")
plt.xlabel('PCA applied BMI and Blood Glucose Level')
plt.ylabel('HbA1c_level')
plt.plot(X_test, predicted, color="orange", linewidth=3)
plt.show()

y_test_array = np.array(y_test)
y_pred_array = np.array(predicted)
print('e', explained_variance_score(y_test, predicted))
