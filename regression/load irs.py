import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)


# Model and Parameter Distribution

class SimpleNN(torch.nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=16):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define parameter distribution for random search
param_dist = {
    'lr': [0.01, 0.001, 0.0001],
    'batch_size': [16, 32, 64],
    'epochs': [50, 100, 150]
}

# Implementing Random Search

from random import choice
from sklearn.metrics import accuracy_score


# Function to train the model
def train_model(model, criterion, optimizer, dataloader):
    model.train()
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# Function to evaluate the model
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    return accuracy_score(all_labels, all_preds)


# Random search
n_iter = 5
best_params = None
best_score = 0
results = []

for _ in range(n_iter):
    params = {key: choice(value) for key, value in param_dist.items()}
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

    model = SimpleNN(input_size=X_tensor.shape[1], num_classes=len(torch.unique(y_tensor)))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # Train the model
    for epoch in range(params['epochs']):
        train_model(model, criterion, optimizer, train_loader)

    # Evaluate the model
    accuracy = evaluate_model(model, test_loader)

    # Store the results
    results.append((params, accuracy))

    if accuracy > best_score:
        best_score = accuracy
        best_params = params

# Print all results
for params, accuracy in results:
    print(f'Params: {params} => Accuracy: {accuracy:.4f}')

print(f'Best parameters: {best_params}')
print(f'Best accuracy: {best_score:.4f}')






