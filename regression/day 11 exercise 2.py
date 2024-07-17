import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load the Wine Quality dataset
red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

# Read the datasets
red_wine = pd.read_csv(red_wine_url, sep=';')
white_wine = pd.read_csv(white_wine_url, sep=';')

# Combine the datasets
data = pd.concat([red_wine, white_wine], ignore_index=True)

# Separate features from the target
X = data.drop('quality', axis=1)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute the hierarchical clustering using the linkage method
Z = linkage(X_scaled, method='ward')

# Plot the dendrogram
plt.figure(figsize=(12, 8))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Apply Agglomerative Clustering to assign cluster labels
model = AgglomerativeClustering(n_clusters=5, linkage='ward')
cluster_labels = model.fit_predict(X_scaled)

# Compute the silhouette score
sil_score = silhouette_score(X_scaled, cluster_labels)
print(f'Silhouette Score: {sil_score:.4f}')
