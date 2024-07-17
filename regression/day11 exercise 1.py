import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(data)

# Convert to PyTorch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)


def kmeans(X, num_clusters, num_iters=100):
    # Randomly initialize cluster centroids
    indices = torch.randperm(X.size(0))[:num_clusters]
    centroids = X[indices]

    for i in range(num_iters):
        # Assign each data point to the nearest centroid
        distances = torch.cdist(X, centroids)
        cluster_assignments = torch.argmin(distances, dim=1)

        # Update centroids
        new_centroids = torch.stack([X[cluster_assignments == k].mean(dim=0) for k in range(num_clusters)])

        # Check for convergence
        if torch.equal(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, cluster_assignments


# Number of clusters
num_clusters = 3

# Run k-Means clustering
centroids, cluster_assignments = kmeans(X_tensor, num_clusters)

# Reduce dimensionality using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the clusters
plt.figure(figsize=(8, 6))
for k in range(num_clusters):
    cluster_data = X_pca[cluster_assignments == k]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {k}')
plt.scatter(pca.transform(centroids)[:, 0], pca.transform(centroids)[:, 1], s=300, c='red', marker='X',
            label='Centroids')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.title('k-Means Clustering on Iris Dataset')
plt.show()

# Compute the silhouette score
sil_score = silhouette_score(X, cluster_assignments.numpy())
print(f'Silhouette Score: {sil_score:.4f}')
