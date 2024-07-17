import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


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


# Define training function with early stopping
def train_with_early_stopping(net, train_loader, val_loader, criterion, optimizer, epochs=100, patience=5):
    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs.view(inputs.size(0), -1))  # Flatten the inputs
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = net(inputs.view(inputs.size(0), -1))  # Flatten the inputs
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))

        early_stopping(val_loss / len(val_loader))

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}/{epochs} - Training Loss: {running_loss / len(train_loader)} - Validation Loss: {val_loss / len(val_loader)}")

        if early_stopping.early_stop:
            print("Early stopping")
            break

    return train_losses, val_losses


# Define function to calculate accuracy
def calculate_accuracy(loader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = net(images.view(images.size(0), -1))  # Flatten the inputs
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# Define transforms for the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False)

# Define network parameters for CIFAR-10 (10 classes)
input_size = 3 * 32 * 32
hidden_size = 100
output_size = 10

# Initialize the network
net = SimpleNN(input_size, hidden_size, output_size)
print(net)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Train the network with early stopping
train_losses, val_losses = train_with_early_stopping(net, trainloader, valloader, criterion, optimizer, epochs=100,
                                                     patience=5)

# Plot the loss curves
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model
train_accuracy = calculate_accuracy(trainloader, net)
val_accuracy = calculate_accuracy(valloader, net)

print(f"Training Accuracy: {train_accuracy}%")
print(f"Validation Accuracy: {val_accuracy}%")

# Save the model
torch.save(net.state_dict(), 'simple_nn_cifar10.pth')

# Load the model
loaded_net = SimpleNN(input_size, hidden_size, output_size)
loaded_net.load_state_dict(torch.load('simple_nn_cifar10.pth'))

# Verify the loaded model
loaded_net.eval()
val_accuracy_loaded = calculate_accuracy(valloader, loaded_net)
print(f"Validation Accuracy of Loaded Model: {val_accuracy_loaded}%")
