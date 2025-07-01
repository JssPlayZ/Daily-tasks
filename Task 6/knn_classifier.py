import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load local CSV
df = pd.read_csv('iris.csv')

# Clean and normalize column names
df.columns = df.columns.str.strip().str.lower()

# Drop 'id' column if present
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# Encode 'species' column
if 'species' not in df.columns:
    raise ValueError("Target column 'species' not found.")

df['species'] = df['species'].astype('category').cat.codes

# Features and target
X = df.drop('species', axis=1)
y = df['species']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Test different K values
accuracy_list = []
k_range = range(1, 21)

for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_list.append(acc)

# Plot Accuracy vs K
plt.figure(figsize=(8, 5))
plt.plot(k_range, accuracy_list, marker='o')
plt.title("Accuracy vs K")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("knn_accuracy_plot.png")
plt.show()

# Use best K
best_k = accuracy_list.index(max(accuracy_list)) + 1
print(f"✅ Best K = {best_k} with accuracy = {max(accuracy_list):.2f}")

# Final Model
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)
y_final_pred = final_model.predict(X_test)

# Evaluation
print("\n📊 Classification Report:")
print(classification_report(y_test, y_final_pred))

print("🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_final_pred))