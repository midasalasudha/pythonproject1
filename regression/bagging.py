# Implementing Bagging
import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim

# Load the dataset
cali = load_iris()
X, y = cali.data, cali.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Function to train a single model
def train_model(model, X_train, y_train, epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()
    return model


# Train multiple models on bootstrapped samples
def bootstrap_sample(X, y):
    indices = torch.randint(0, len(X), (len(X),))
    return X[indices], y[indices]


num_models = 2
models = []
for _ in range(num_models):
    X_bootstrap, y_bootstrap = bootstrap_sample(X_train, y_train)
    model = Net()
    model = train_model(model, X_bootstrap, y_bootstrap)
    models.append(model)


# Function to make predictions with an ensemble of models
def predict(models, X):
    predictions = [model(X).detach().numpy() for model in models]
    avg_predictions = np.mean(predictions, axis=0)
    return torch.tensor(avg_predictions)


# Make predictions on the test set
y_pred = predict(models, X_test)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# Define a simple neural network for classification
class ClassifierNet(nn.Module):
    def __init__(self):
        super(ClassifierNet, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Function to train a single model
def train_model(model, X_train, y_train, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    return model


# Train multiple models on bootstrapped samples
num_models = 2
models = []
for _ in range(num_models):
    X_bootstrap, y_bootstrap = bootstrap_sample(X_train, y_train)
    model = ClassifierNet()
    model = train_model(model, X_bootstrap, y_bootstrap)
    models.append(model)


# Function to make predictions with an ensemble of models
def predict(models, X):
    predictions = [model(X).detach().numpy() for model in models]
    avg_predictions = np.mean(predictions, axis=0)
    return torch.tensor(np.argmax(avg_predictions, axis=1))


# Make predictions on the test set
y_pred = predict(models, X_test)

from sklearn.metrics import accuracy_score, classification_report

# Evaluate the ensemble model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")






