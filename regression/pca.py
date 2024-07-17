# PCA with PyTorch

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Preparing the Dataset
# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float()


# Computing the Covariance Matrix
# Standardize the dataset
X_mean = X_tensor.mean(dim=0)
X_std = X_tensor.std(dim=0)
X_standardized = (X_tensor - X_mean) / X_std
#print('Mean, std & Stand ', X_mean, X_std, X_standardized)

# Compute the covariance matrix
cov_matrix = torch.mm(X_standardized.T, X_standardized) / (X_standardized.size(0) - 1)
print('Cov Mat \n',cov_matrix)

# Performing Eigen Decomposition
# Perform eigen decomposition using torch.linalg.eig
eigenvalues_complex, eigenvectors_complex = torch.linalg.eig(cov_matrix)
print('Eig val, Eig vec ', eigenvalues_complex, eigenvectors_complex)

# Extract the real part of the eigenvalues and eigenvectors
eigenvalues = eigenvalues_complex.real
eigenvectors = eigenvectors_complex.real

# Projecting Data onto the New Feature Space
# Sort eigenvalues and corresponding eigenvectors
sorted_indices = torch.argsort(eigenvalues, descending=True)
sorted_eigenvectors = eigenvectors[:, sorted_indices]
print('Sorted EVec', sorted_eigenvectors)

# Select the top k eigenvectors
k = 2
top_k_eigenvectors = sorted_eigenvectors[:, :k]

# Project the data onto the new feature space
X_reduced = torch.mm(X_standardized, top_k_eigenvectors)

# Visualizing the Reduced Dimensionality Data

# Convert to NumPy for plotting
X_reduced_np = X_reduced.detach().numpy()
y_np = y

# Plot the reduced dimensionality data
plt.figure(figsize=(8, 6))
for target in np.unique(y_np):
    indices = y_np == target
    plt.scatter(X_reduced_np[indices, 0], X_reduced_np[indices, 1], label=data.target_names[target])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.legend()
plt.show()



