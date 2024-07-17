import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Preparing the Dataset
# Generate some example data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 5 * X + 2

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float()


# Defining the Model

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearRegressionModel()

# Training the Model

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluating the Model

model.eval()
with torch.no_grad():
    predicted = model(X_tensor).detach().numpy()

# Plotting the results
plt.scatter(X, y, label='Original data')
plt.plot(X, predicted, label='Fitted line', color='red')
plt.legend()
plt.show()
