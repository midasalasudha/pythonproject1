# Implementing LDA with PyTorch

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
y_tensor = torch.from_numpy(y).long()


# Computing Scatter Matrices
# Compute the overall mean
mean_overall = X_tensor.mean(dim=0)

# Compute the within-class scatter matrix
classes = torch.unique(y_tensor)
print('Classes :', classes)

S_W = torch.zeros((X_tensor.size(1), X_tensor.size(1)))
for c in classes:
    X_c = X_tensor[y_tensor == c]
    mean_c = X_c.mean(dim=0)
    S_W += torch.mm((X_c - mean_c).T, (X_c - mean_c))

# Compute the between-class scatter matrix
S_B = torch.zeros((X_tensor.size(1), X_tensor.size(1)))
for c in classes:
    N_c = X_tensor[y_tensor == c].size(0)
    mean_c = X_tensor[y_tensor == c].mean(dim=0)
    mean_diff = (mean_c - mean_overall).unsqueeze(1)
    S_B += N_c * torch.mm(mean_diff, mean_diff.T)


# Performing Eigen Decomposition

# Compute the matrix S_W^{-1} S_B
eigvals, eigvecs = torch.linalg.eig(torch.mm(torch.linalg.inv(S_W), S_B))

# Extract the real part of the eigenvalues and eigenvectors
eigvals = eigvals.real
eigvecs = eigvecs.real


# Projecting Data onto the New Feature Space

# Sort eigenvalues and corresponding eigenvectors
sorted_indices = torch.argsort(eigvals, descending=True)
sorted_eigvecs = eigvecs[:, sorted_indices]

# Select the top k eigenvectors
k = 2
top_k_eigvecs = sorted_eigvecs[:, :k]

# Project the data onto the new feature space
X_lda = torch.mm(X_tensor, top_k_eigvecs)


# Visualizing the Reduced Dimensionality Data

# Convert to NumPy for plotting
X_lda_np = X_lda.detach().numpy()
y_np = y

# Plot the reduced dimensionality data
plt.figure(figsize=(8, 6))
for target in np.unique(y_np):
    indices = y_np == target
    plt.scatter(X_lda_np[indices, 0], X_lda_np[indices, 1], label=data.target_names[target])
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('LDA of Iris Dataset')
plt.legend()
plt.show()


