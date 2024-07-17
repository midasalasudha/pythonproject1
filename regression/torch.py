import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Generate some example data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.2

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float()

print('Ridge Regression Epochs')


class RidgeRegressionModel(nn.Module):
    def __init__(self):
        super(RidgeRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = RidgeRegressionModel()

# Define the loss function with L2 regularization
criterion = nn.MSELoss()
lambda_reg = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Add L2 regularization
    l2_reg = torch.tensor(0.)
    for param in model.parameters():
        l2_reg += torch.linalg.norm(param, 2)
    loss += lambda_reg * l2_reg

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# ===============================================================

print('Lasso Regression Epochs')


class LassoRegressionModel(nn.Module):
    def __init__(self):
        super(LassoRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = LassoRegressionModel()

# Define the loss function with L1 regularization
criterion = nn.MSELoss()
lambda_reg = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Add L1 regularization
    l1_reg = torch.tensor(0.)
    for param in model.parameters():
        l1_reg += torch.linalg.norm(param, 1)
    loss += lambda_reg * l1_reg

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


def from_numpy(X):
    return None