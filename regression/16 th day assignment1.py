# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.output(x)
        return x


# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# Define the dataset
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# Define network parameters
input_size = X_train.shape[1]
hidden_size = 5
output_size = 1

# Initialize the network
net = SimpleNN(input_size, hidden_size, output_size)
print(net)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)


# Training function with early stopping
def train_with_early_stopping(net, X_train, y_train, X_val, y_val, criterion, optimizer, epochs=100, batch_size=32,
                              patience=5):
    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(epochs):
        net.train()
        permutation = torch.randperm(X_train.size(0))

        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            outputs = net(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item())

        net.eval()
        with torch.no_grad():
            val_outputs = net(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())

        early_stopping(val_loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Training Loss: {loss.item()} - Validation Loss: {val_loss.item()}")

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return train_losses, val_losses


# Train the network with early stopping
train_losses, val_losses = train_with_early_stopping(net, X_train, y_train, X_val, y_val, criterion, optimizer)

# Plot the loss curves
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Create a sample input tensor for forward propagation
input_tensor = torch.randn(1, input_size)

# Forward propagation
output = net(input_tensor)
print("Output:", output)
