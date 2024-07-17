# t-SNE with PyTorch

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

# Defining Parameters (Perplexity, Learning Rate)
perplexity = 7
learning_rate = 200
num_iterations = 1000

# Running the t-SNE Algorithm

from sklearn.manifold import TSNE

# Run t-SNE
tsne = TSNE(n_components=2, perplexity=perplexity,
            learning_rate=learning_rate, n_iter=num_iterations, random_state=42)
X_tsne = tsne.fit_transform(X)

# Convert to PyTorch tensor
X_tsne_tensor = torch.from_numpy(X_tsne).float()

# Visualizing the Reduced Dimensionality Data
# Plot the reduced dimensionality data
plt.figure(figsize=(8, 6))
for target in np.unique(y):
    indices = y == target
    plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=data.target_names[target])
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('t-SNE of Iris Dataset')
plt.legend()
plt.show()



