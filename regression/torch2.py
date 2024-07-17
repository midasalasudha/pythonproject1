import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Calculating MAE
# Load the California housing dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float()

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor,
                                                    test_size=0.2, random_state=42)


# Define a simple neural network model for regression
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)


# Function to calculate Mean Absolute Error (MAE)
def calculate_mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred)).item()


# Initialize model, loss function, and optimizer
model = SimpleNN(input_size=X_tensor.shape[1])
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create dataloaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Train the model
model.train()
for epoch in range(100):  # Train for 100 epochs
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
y_pred = []
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs)
        y_pred.append(outputs)
y_pred = torch.cat(y_pred, dim=0)

# Calculate MAE
mae = calculate_mae(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae:.4f}')


# Calculating MSE and RMSE

# Function to calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
def calculate_mse_rmse(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2).item()
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
    return mse, rmse


# Calculate MSE and RMSE
mse, rmse = calculate_mse_rmse(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
