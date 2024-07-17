
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Preparing the Dataset
# Generate synthetic data
n_samples = 30
n_features = 2
n_clusters = 3
X, y = make_blobs(n_samples=n_samples, n_features=n_features,
                  centers=n_clusters)
print('X values : \n ', X, '\n y values : \n ', y)
# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float()


# Defining and Initializing Centroids
def initialize_centroids(X, k):
    indices = torch.randperm(X.size(0))[:k]
    return X[indices]


k = 3
'''centroids = initialize_centroids(X_tensor, k)
print(f'Initial centroids:\n{centroids}')
'''


# Performing the Assignment and Update Steps
def assign_clusters(X, centroids):
    distances = torch.cdist(X, centroids)
    return torch.argmin(distances, dim=1)


def update_centroids(X, labels, k):
    new_centroids = torch.zeros((k, X.size(1)))
    for i in range(k):
        points = X[labels == i]
        new_centroids[i] = points.mean(dim=0)
    return new_centroids


def kmeans(X, k, max_iters=10, tol=1e-4):
    global labels
    centroids = initialize_centroids(X, k)
    print(f'Centroids:\n{centroids}')
    for i in range(max_iters):
        labels = assign_clusters(X, centroids)
        print('Labels \n', labels)
        new_centroids = update_centroids(X, labels, k)
        if torch.all(torch.norm(new_centroids - centroids, dim=1) < tol):
            break
        centroids = new_centroids
    return centroids, labels


# Run k-means clustering
final_centroids, final_labels = kmeans(X_tensor, k)
# Evaluating the Clusters
# Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=final_labels.numpy(), cmap='viridis', marker='o', edgecolor='k', s=50)
plt.scatter(final_centroids[:, 0].numpy(), final_centroids[:, 1].numpy(), c='red', marker='x', s=200)
plt.title('k-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


# Within-cluster sum of squares (WCSS)
def calculate_wcss(X, labels, centroids):
    wcss = 0
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        wcss += torch.sum((cluster_points - centroids[i]) ** 2).item()
    return wcss


wcss = calculate_wcss(X_tensor, final_labels, final_centroids)
print(f'Within-cluster sum of squares: {wcss}')

like
3