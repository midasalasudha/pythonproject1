import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram

# Generate synthetic data
n_samples = 30
n_features = 2
X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=3)

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float()


# Define distance function
def euclidean_distance(a, b):
    return torch.sqrt(torch.sum((a - b) ** 2))


def pairwise_distances(X):
    n = X.size(0)
    distances = torch.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = euclidean_distance(X[i], X[j])
            distances[j, i] = distances[i, j]
    return distances


# Perform agglomerative clustering
def agglomerative_clustering(X, method='single'):
    n = X.size(0)
    distances = pairwise_distances(X)
    clusters = {i: [i] for i in range(n)}
    history = []

    cluster_mapping = {i: i for i in range(n)}

    while len(clusters) > 1:
        min_dist = float('inf')
        to_merge = None

        cluster_keys = list(clusters.keys())
        for i in range(len(cluster_keys)):
            for j in range(i + 1, len(cluster_keys)):
                key_i = cluster_keys[i]
                key_j = cluster_keys[j]
                if method == 'single':
                    dist = torch.min(distances[clusters[key_i], :][:, clusters[key_j]])
                elif method == 'complete':
                    dist = torch.max(distances[clusters[key_i], :][:, clusters[key_j]])
                elif method == 'average':
                    dist = torch.mean(distances[clusters[key_i], :][:, clusters[key_j]])
                if dist < min_dist:
                    min_dist = dist
                    to_merge = (key_i, key_j)

        i, j = to_merge
        clusters[i].extend(clusters[j])
        del clusters[j]

        history.append([cluster_mapping[i], cluster_mapping[j], min_dist.item(), len(clusters[i])])

        for k in clusters:
            if k != i:
                if method == 'single':
                    dist = torch.min(distances[clusters[i], :][:, clusters[k]])
                elif method == 'complete':
                    dist = torch.max(distances[clusters[i], :][:, clusters[k]])
                elif method == 'average':
                    dist = torch.mean(distances[clusters[i], :][:, clusters[k]])
                distances[i, k] = dist
                distances[k, i] = dist

        cluster_mapping[i] = n
        n += 1

    return history


# Perform agglomerative clustering
history = agglomerative_clustering(X_tensor, method='single')


# Correcting the linkage matrix format
def format_history(history):
    n = len(history)
    new_history = []
    for i, (a, b, dist, count) in enumerate(history):
        new_history.append([a, b, dist, count])
    return np.array(new_history)


# Format history for dendrogram
formatted_history = format_history(history)


# Plot dendrogram
def plot_dendrogram(history):
    plt.figure(figsize=(10, 7))
    dendrogram(history, leaf_rotation=90, leaf_font_size=10)
    plt.title('Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()


plot_dendrogram(formatted_history)
