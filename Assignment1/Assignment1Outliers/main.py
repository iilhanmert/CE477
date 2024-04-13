import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"C:\Users\MERT\Downloads\diabetes_prediction_dataset.csv")
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]


outliers_age = detect_outliers(df, 'age')
outliers_bmi = detect_outliers(df, 'bmi')
outliers_HbA1c = detect_outliers(df, 'HbA1c_level')
outliers_glucose = detect_outliers(df, 'blood_glucose_level')

print("Outliers in 'age':", outliers_age)
print("Outliers in 'bmi':", outliers_bmi)
print("Outliers in 'HbA1c_level':", outliers_HbA1c)
print("Outliers in 'blood_glucose_level':", outliers_glucose)


def plot_outliers(df, outliers_data, col, title):
    plt.figure(figsize=(5, 4))
    sns.boxplot(y=df[col], color='lightblue', showfliers=False)  # Dahili aykırı değerleri çıkarma
    sns.scatterplot(x=[1] * len(outliers_data), y=outliers_data[col], color='red', s=50, edgecolor='black')
    plt.title(title)
    plt.show()


plot_outliers(df, outliers_age, 'age', 'Outliers in age')
plot_outliers(df, outliers_bmi, 'bmi', 'Outliers in bmi')
plot_outliers(df, outliers_HbA1c, 'HbA1c_level', 'Outliers in HbA1c_level')
plot_outliers(df, outliers_glucose, 'blood_glucose_level', 'Outliers in blood_glucose_level')
