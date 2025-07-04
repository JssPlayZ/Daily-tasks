import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv('Mall_Customers.csv')

# Optional: drop customer ID
df.drop('CustomerID', axis=1, inplace=True)

# Select features (e.g., Annual Income and Spending Score)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- Elbow Method ----------
inertia = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# Plot Elbow
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title("Elbow Method - Optimal K")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.savefig("elbow_method.png")
plt.show()

# ---------- Fit KMeans ----------
optimal_k = 5  # Based on elbow curve
model = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
y_kmeans = model.fit_predict(X_scaled)

# Add clusters to original data
df['Cluster'] = y_kmeans

# ---------- Cluster Plot ----------
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=y_kmeans, palette='Set2', s=80)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.title(f'KMeans Clustering (K={optimal_k})')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.legend()
plt.tight_layout()
plt.savefig("cluster_plot.png")
plt.show()

# ---------- Evaluate with Silhouette Score ----------
score = silhouette_score(X_scaled, y_kmeans)
print(f"✅ Silhouette Score for K={optimal_k}: {score:.3f}")
