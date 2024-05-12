import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn import preprocessing as preproc

# Load the dataset
data = pd.read_csv("/Users/canbaytekin/Documents/GitHub/CE477/data/diabetes_prediction_dataset.csv")
df = pd.DataFrame(data)

dataColumn = ['bmi', 'blood_glucose_level']

# Extract features and target variable
X = df[["bmi", "blood_glucose_level"]].values
y = df['HbA1c_level'].values

# Normalize the features
scaler = preproc.MinMaxScaler()
normalize = scaler.fit_transform(X)
normalized_df = pd.DataFrame(normalize, columns=dataColumn)
print(normalized_df)

# Apply PCA
pca = PCA(n_components=1)
X = pca.fit_transform(X)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train a linear regression model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
predicted = regr.predict(X_test)

# Print coefficients, RMSE, and R^2 score
print()
print("Coefficients: ", regr.coef_)
print("Coefficient of determination (R^2): %.2f" % r2_score(y_test, predicted))
print("Root Mean Squared Error (RMSE): %.2f" % mean_squared_error(y_test, predicted))

# Plot the results
plt.scatter(X_test, y_test, color="black")
plt.xlabel('PCA applied BMI and Blood Glucose Level')
plt.ylabel('HbA1c_level')
plt.plot(X_test, predicted, color="orange", linewidth=3)
plt.show()

# Calculate and print MAPE
absolute_errors = np.abs((y_test - predicted) / y_test)
mape = np.mean(absolute_errors) * 100
print('Mean Absolute Percentage Error (MAPE): %.2f%%' % mape)
