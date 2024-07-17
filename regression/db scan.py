# DBSCAN with PyTorch

import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate synthetic data
n_samples = 30
n_features = 2
X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=3, random_state=42)

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float()


# Define distance function
def euclidean_distance(a, b):
    return torch.sqrt(torch.sum((a - b) ** 2, dim=1))


epsilon = 0.5
minPts = 5


# DBSCAN functions
def region_query(X, point_idx, epsilon):
    distances = euclidean_distance(X[point_idx].unsqueeze(0), X)
    return torch.where(distances <= epsilon)[0]


def expand_cluster(X, labels, point_idx, cluster_id, epsilon, minPts):
    seeds = region_query(X, point_idx, epsilon)
    if len(seeds) < minPts:
        labels[point_idx] = -1  # Mark as noise
        return False
    else:
        labels[seeds] = cluster_id
        seeds = seeds[seeds != point_idx]

        while len(seeds) > 0:
            current_point = seeds[0]
            results = region_query(X, current_point, epsilon)
            if len(results) >= minPts:
                for i in results:
                    if labels[i] == 0:
                        seeds = torch.cat((seeds, torch.tensor([i])))
                        labels[i] = cluster_id
                    elif labels[i] == -1:
                        labels[i] = cluster_id
            seeds = seeds[1:]
        return True


def dbscan(X, epsilon, minPts):
    cluster_id = 0
    labels = torch.zeros(X.size(0), dtype=torch.int)

    for point_idx in range(X.size(0)):
        if labels[point_idx] == 0:
            if expand_cluster(X, labels, point_idx, cluster_id + 1, epsilon, minPts):
                cluster_id += 1

    return labels


# Perform DBSCAN clustering
labels = dbscan(X_tensor, epsilon, minPts)


# Plot the clusters
def plot_clusters(X, labels):
    unique_labels = torch.unique(labels)
    colors = plt.get_cmap("viridis", len(unique_labels))
    plt.figure(figsize=(8, 6))
    for k in unique_labels:
        if k == -1:
            color = 'k'
            marker = 'x'
        else:
            color = colors(k / len(unique_labels))
            marker = 'o'
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[color], marker=marker, s=50, label=f'Cluster {k}' if k != -1 else 'Noise')
    plt.title('DBSCAN Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


plot_clusters(X_tensor.numpy(), labels)



