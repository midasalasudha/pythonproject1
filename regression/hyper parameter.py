# Hyperparameter Tuning with Grid Search
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target
# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()
# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor,
                                                    test_size=0.2, random_state=42)


# Define a simple neural network model for classification
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=16):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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


# Define parameter grid for hyperparameter tuning
param_grid = {
    'batch_size': [16, 32],
    'epochs': [50, 100],
    'lr': [0.01, 0.001]
}
best_params = None
best_score = 0
# Perform grid search
for params in ParameterGrid(param_grid):
    print(f' Params :  {params}')
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
    print(f'Accuracy : {accuracy}')
    if accuracy > best_score:
        best_score = accuracy
        best_params = params
print(f'Best parameters: {best_params}')
print(f'Best accuracy: {best_score:.4f}')

