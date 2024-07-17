# Nested Cross - Validation

import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import ParameterGrid

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()


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
    'hidden_size': [8, 16, 32],
    'lr': [0.01, 0.001]
}

# Nested cross-validation
outer_k = 5
inner_k = 3
outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=inner_k, shuffle=True, random_state=42)

outer_accuracies = []

for train_idx, test_idx in outer_cv.split(X_tensor):
    X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
    y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]

    best_params = None
    best_score = 0

    for params in ParameterGrid(param_grid):
        inner_scores = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train):
            X_inner_train, X_inner_val = X_train[inner_train_idx], X_train[inner_val_idx]
            y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]

            train_dataset = TensorDataset(X_inner_train, y_inner_train)
            val_dataset = TensorDataset(X_inner_val, y_inner_val)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

            model = SimpleNN(input_size=X_tensor.shape[1],
                             num_classes=len(torch.unique(y_tensor)),
                             hidden_size=params['hidden_size'])
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

            train_model(model, criterion, optimizer, train_loader)
            inner_score = evaluate_model(model, val_loader)
            inner_scores.append(inner_score)

        avg_inner_score = sum(inner_scores) / inner_k
        if avg_inner_score > best_score:
            best_score = avg_inner_score
            best_params = params

    # Train on the entire training set with the best hyperparameters
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = SimpleNN(input_size=X_tensor.shape[1], num_classes=len(torch.unique(y_tensor)),
                     hidden_size=best_params['hidden_size'])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])

    train_model(model, criterion, optimizer, train_loader)
    outer_score = evaluate_model(model, test_loader)
    outer_accuracies.append(outer_score)

    print(f'Outer fold accuracy: {outer_score:.4f}')

# Average accuracy across all outer folds
average_outer_accuracy = sum(outer_accuracies) / outer_k
print(f'Average outer fold accuracy: {average_outer_accuracy:.4f}')






