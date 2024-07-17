# Polynomial Regression with PyTorch
# Preparing the Dataset

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# Generate some example data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 * X ** 2 + 3 * X + 1 + np.random.randn(100, 1) * 0.2

# Convert to PyTorch tensors
# X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float()

# Feature Engineering (Creating Polynomial Features)

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Convert polynomial features to PyTorch tensor
X_poly_tensor = torch.from_numpy(X_poly).float()


# Defining and Training the Model

class PolynomialRegressionModel(nn.Module):
    def __init__(self):
        super(PolynomialRegressionModel, self).__init__()
        self.linear = nn.Linear(X_poly.shape[1], 1)  # Adjust input size to match polynomial features

    def forward(self, x):
        return self.linear(x)


model = PolynomialRegressionModel()

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_poly_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluating the Model

model.eval()
with torch.no_grad():
    predicted = model(X_poly_tensor).detach().numpy()

# Plotting the results
plt.scatter(X, y, label='Original data')
plt.scatter(X, predicted, label='Fitted curve', color='red')
plt.legend()
plt.show()


