import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('./Titanic-Dataset.csv')

# ------------------ Summary Statistics ------------------
print("🔹 Summary Statistics:")
print(df.describe(include='all'))

# ------------------ Data Types and Nulls ------------------
print("\n🔹 Data Types and Null Values:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

# ------------------ Histograms ------------------
plt.figure(figsize=(12, 5))
df['Age'].hist(bins=30, edgecolor='black')
plt.title("Histogram of Age")
plt.xlabel("Age")
plt.ylabel("Count")
plt.grid(False)
plt.tight_layout()
plt.savefig("age_histogram.png")
plt.show()

plt.figure(figsize=(12, 5))
df['Fare'].hist(bins=30, edgecolor='black', color='orange')
plt.title("Histogram of Fare")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.grid(False)
plt.tight_layout()
plt.savefig("fare_histogram.png")
plt.show()

# ------------------ Boxplots ------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Boxplot of Age by Survival")
plt.savefig("boxplot_age_survived.png")
plt.show()

# ------------------ Correlation Heatmap ------------------
# Drop non-numeric and irrelevant columns
df_corr = df.copy()
df_corr = df_corr.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])

# If 'Sex' or 'Embarked' are still strings, encode them:
df_corr['Sex'] = df_corr['Sex'].map({'male': 0, 'female': 1})
df_corr['Embarked'] = df_corr['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

df_corr = df.copy()
df_corr['Sex'] = df_corr['Sex'].map({'male': 0, 'female': 1})
df_corr['Embarked'] = df_corr['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

plt.figure(figsize=(10, 6))
sns.pairplot(df_corr[['Survived', 'Age', 'Fare', 'Pclass', 'Sex']].dropna(), hue='Survived')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# ------------------ Pairplot ------------------
sns.pairplot(df_corr[['Survived', 'Age', 'Fare', 'Pclass', 'Sex']].dropna(), hue='Survived')
plt.savefig("pairplot.png")
plt.show()

# ------------------ Countplot for Categorical Features ------------------
plt.figure(figsize=(8, 5))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Passenger Class vs Survival")
plt.savefig("countplot_pclass.png")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Gender vs Survival")
plt.savefig("countplot_sex.png")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='Embarked', hue='Survived', data=df)
plt.title("Embarked Port vs Survival")
plt.savefig("countplot_embarked.png")
plt.show()