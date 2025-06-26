import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('Housing.csv')

# Encode categorical column 'furnishingstatus'
df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df.drop('price', axis=1)
y = df['price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)
print("Model Coefficients:\n", pd.Series(model.coef_, index=X.columns))

# Regression Plot (using one feature for visual simplicity)
plt.figure(figsize=(8, 5))
sns.regplot(x=X_test['area'], y=y_test, label='Actual', scatter_kws={"color": "blue"})
sns.lineplot(x=X_test['area'], y=y_pred, color='red', label='Predicted')
plt.title('Linear Regression: Area vs Price')
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend()
plt.savefig("regression_plot.png")
plt.show()