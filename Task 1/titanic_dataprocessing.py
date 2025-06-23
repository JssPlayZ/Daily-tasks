# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset to process it
df = pd.read_csv('./Titanic-Dataset.csv')
print(df.head())
print(df.info())

# here we check for the missing values in that dataset if any
print("\nMissing values:\n", df.isnull().sum())

# Handling the missing values accordingly
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Cabin'].fillna('Unknown', inplace=True)  # Optional: drop or simplify later

# dropping the columns that we don't need
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# Encoding the categorical features
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# normalizing numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Detecting and Plotting/Mapping Outliers
sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplot of Age & Fare")
plt.show()
plt.savefig("boxplot.png")

# Removing outliers using IQR Q3 - Q1
Q1 = df[['Age', 'Fare']].quantile(0.25)
Q3 = df[['Age', 'Fare']].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[['Age', 'Fare']] < (Q1 - 1.5 * IQR)) | (df[['Age', 'Fare']] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Overview of the data
print("\nCleaned Dataset:\n", df.head())
print(df.describe())
print(df.info())