import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
wine_data = load_wine()
df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
k = 3
kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)
df['cluster_assignment'] = cluster_labels
sil_score = silhouette_score(X_scaled, cluster_labels)
print("=== K-Means Clustering Results (Wine Dataset) ===")
print(f"Number of Clusters (k): {k}")
print(f"Silhouette Score: {sil_score:.4f}\n")
print("First 5 rows of data with assignments:")
print(df.head())
