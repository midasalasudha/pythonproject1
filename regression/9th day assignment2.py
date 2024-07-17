import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Load the dataset
boston = load_boston()
data = boston.data
target = boston.target

# Standardize the data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


# Train Linear Regression Model
def train_model(X_train, y_train, X_test, y_test, input_dim, learning_rate=0.01, num_epochs=1000):
    model = LinearRegressionModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_predictions = model(X_train).numpy()
        test_predictions = model(X_test).numpy()

    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)

    return train_predictions, test_predictions, train_mse, test_mse, model


# Polynomial Regression Model
def polynomial_regression(degree, X_train, X_test):
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    return X_train_poly, X_test_poly, X_train_poly.shape[1]


# Train and evaluate linear regression model
train_pred_linear, test_pred_linear, train_mse_linear, test_mse_linear, _ = train_model(X_train_tensor, y_train_tensor,
                                                                                        X_test_tensor, y_test_tensor,
                                                                                        X_train_tensor.shape[1])

# Polynomial regression with degree 3
degree = 3
X_train_poly, X_test_poly, input_dim_poly = polynomial_regression(degree, X_train, X_test)

X_train_poly_tensor = torch.tensor(X_train_poly, dtype=torch.float32)
X_test_poly_tensor = torch.tensor(X_test_poly, dtype=torch.float32)

# Train and evaluate polynomial regression model
train_pred_poly, test_pred_poly, train_mse_poly, test_mse_poly, _ = train_model(X_train_poly_tensor, y_train_tensor,
                                                                                X_test_poly_tensor, y_test_tensor,
                                                                                input_dim_poly)

# Compare MSE
print(f"Linear Regression - Train MSE: {train_mse_linear:.4f}, Test MSE: {test_mse_linear:.4f}")
print(f"Polynomial Regression (degree {degree}) - Train MSE: {train_mse_poly:.4f}, Test MSE: {test_mse_poly:.4f}")

# Plot predicted vs actual values for linear regression (training set)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_train, train_pred_linear, color='blue', label='Predicted')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2, label='Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression - Training Set')
plt.legend()

# Plot predicted vs actual values for polynomial regression (training set)
plt.subplot(1, 2, 2)
plt.scatter(y_train, train_pred_poly, color='blue', label='Predicted')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2, label='Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Polynomial Regression - Training Set')
plt.legend()

plt.show()

# Plot predicted vs actual values for linear regression (test set)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, test_pred_linear, color='red', label='Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression - Test Set')
plt.legend()

# Plot predicted vs actual values for polynomial regression (test set)
plt.subplot(1, 2, 2)
plt.scatter(y_test, test_pred_poly, color='red', label='Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Polynomial Regression - Test Set')
plt.legend()

plt.show()
