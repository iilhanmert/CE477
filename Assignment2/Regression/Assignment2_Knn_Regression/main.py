import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import explained_variance_score

data = pd.read_csv("/Users/canbaytekin/Documents/GitHub/CE477/data/diabetes_prediction_dataset.csv")

df = pd.DataFrame(data)

dataColumn = ['bmi', 'blood_glucose_level']

X = df[["bmi", "blood_glucose_level"]].values
y = df['HbA1c_level'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.size)
print(y_train.size)

rmse_val = []
for K in range(50):
    K = K + 1
    model = neighbors.KNeighborsRegressor(n_neighbors=K)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    error = sqrt(mean_squared_error(y_test, pred))
    rmse_val.append(error)

    print('RMSE value for k= ', K, 'is:', error)

curve = pd.DataFrame(rmse_val)
plt.plot(curve)
plt.title('Elbow Curve')
plt.xlabel('k-value')
plt.ylabel('Error (RMSE)')
plt.show()

k_range = range(1, 26)
testing_scores = []
training_scores = []
for k in k_range:
    knn = neighbors.KNeighborsRegressor(n_neighbors=k, weights='uniform', algorithm='auto')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_test = y_test.reshape(-1, 1)
    testing_scores.append(knn.score(X_train, y_train))
    training_scores.append(knn.score(X_test, y_test))

print(testing_scores)
plt.plot(k_range, testing_scores, label='testing score')
plt.plot(k_range, training_scores, label='training score')
plt.legend()
plt.xlabel('Value of K for KNN')
plt.ylabel('R-square Score')
print('Explained Variance Score: ', explained_variance_score(y_test, y_pred))
plt.show()
