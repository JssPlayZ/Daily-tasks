import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv('breast_cancer.csv')

# Drop unwanted columns
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True, errors='ignore')

# Encode target
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------- SVM Linear ----------
svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)

print("\n🔹 SVM Linear Kernel")
print(confusion_matrix(y_test, y_pred_linear))
print(classification_report(y_test, y_pred_linear))

# ---------- SVM RBF ----------
svm_rbf = SVC(kernel='rbf', C=1, gamma=0.1)
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)

print("\n🔹 SVM RBF Kernel")
print(confusion_matrix(y_test, y_pred_rbf))
print(classification_report(y_test, y_pred_rbf))

# ---------- PCA for Visualization ----------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Redo split on PCA data
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Refit models on PCA-transformed data
svm_linear_pca = SVC(kernel='linear', C=1)
svm_linear_pca.fit(X_train_pca, y_train_pca)

svm_rbf_pca = SVC(kernel='rbf', C=1, gamma=0.1)
svm_rbf_pca.fit(X_train_pca, y_train_pca)

# Decision boundary plot function
def plot_decision_boundary(model, X, y, title, filename):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Visualize
plot_decision_boundary(svm_linear_pca, X_pca, y, "SVM Decision Boundary - Linear", "svm_decision_boundary_linear.png")
plot_decision_boundary(svm_rbf_pca, X_pca, y, "SVM Decision Boundary - RBF", "svm_decision_boundary_rbf.png")