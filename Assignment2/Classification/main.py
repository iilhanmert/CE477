import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

file_path = r"C:\Users\MERT\PycharmProjects\DataScience\Assignment2\diabetes_prediction_dataset.csv"
df = pd.read_csv(file_path)

# One-hot encoding for categorical variables
X_encoded = pd.get_dummies(df.drop(columns=['diabetes']), columns=['gender', 'smoking_history', 'hypertension', 'heart_disease'])

# Our target variable for classification
y = df['diabetes']

# Splitting the data into training(%80) and testing(%20) sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Normalize the data for KNN and Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Decision Tree Classifier Training
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_scaled, y_train)

# Logistic Regression Classifier Training
lr_classifier = LogisticRegression(max_iter=1000, random_state=42)
lr_classifier.fit(X_train_scaled, y_train)

# k-Nearest Neighbors Classifier Training
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_scaled, y_train)


classifiers = {
    'Decision Tree': dt_classifier,
    'Logistic Regression': lr_classifier,
    'k-Nearest Neighbors': knn_classifier,

}

for name, clf in classifiers.items():
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"Classification Report for {name}:\n{classification_report(y_test, y_pred)}\n")


# Visualizing the learned decision tree
plt.figure(figsize=(12, 8))
plot_tree(dt_classifier, filled=True, feature_names=X_encoded.columns, class_names=['Non-Diabetic', 'Diabetic'], rounded=True, max_depth=3)
plt.title("Decision Tree Visualization")
plt.savefig('decision_tree.pdf', format='pdf')
plt.show()



