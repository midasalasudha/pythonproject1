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
losses = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    # Add L2 regularization
    l2_reg = torch.tensor(0.)
    for param in model.parameters():
        l2_reg += torch.norm(param, 2)
    loss += lambda_reg * l2_reg
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
# Evaluation and Visualization
model.eval()
with torch.no_grad():
    predicted = model(X_tensor).detach().numpy()
# Plotting the results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.subplot(1, 2, 2)
plt.scatter(X, y, label='Original data')
plt.plot(X, predicted, label='Fitted line', color='red')
plt.legend()
plt.title('Ridge Regression Fit')
plt.show()


# ===============================================================

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
losses = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    # Add L1 regularization
    l1_reg = torch.tensor(0.)
    for param in model.parameters():
        l1_reg += torch.norm(param, 1)
    loss += lambda_reg * l1_reg
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
# Evaluation and Visualization
model.eval()
with torch.no_grad():
    predicted = model(X_tensor).detach().numpy()
# Plotting the results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.subplot(1, 2, 2)
plt.scatter(X, y, label='Original data')
plt.plot(X, predicted, label='Fitted line', color='red')
plt.legend()
plt.title('Lasso Regression Fit')
plt.show()


# ========================================================

def calculate_metrics(y_true, y_pred):
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    mse = torch.mean((y_true - y_pred) ** 2).item()
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot).item()
    return mae, mse, rmse, r2


# Calculate metrics for Ridge Regression
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor)
    mae, mse, rmse, r2 = calculate_metrics(y_tensor, y_pred)
    print(f'Ridge Regression - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R-squared: {r2:.4f}')
# Calculate metrics for Lasso Regression
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor)
    mae, mse, rmse, r2 = calculate_metrics(y_tensor, y_pred)
    print(f'Lasso Regression - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R-squared: {r2:.4f}')

