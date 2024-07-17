import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
from ray.air import session
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)


# Define a simple neural network model for classification
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the training function to be used by Hyperband
def train_model(config):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=int(config["batch_size"]), shuffle=False)

    model = SimpleNN(input_size=X_tensor.shape[1], num_classes=len(torch.unique(y_tensor)),
                     hidden_size=int(config["hidden_size"]))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(int(config["epochs"])):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate the model at the end of each epoch
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        session.report({"mean_accuracy": accuracy})


# Define the search space for Hyperband
search_space = {
    "hidden_size": tune.choice([8, 16, 32]),
    "lr": tune.loguniform(1e-4, 1e-2),
    "batch_size": tune.choice([16, 32, 64]),
    "epochs": tune.choice([5, 10, 15])
}

# Set up Hyperband scheduler
scheduler = HyperBandScheduler(metric="mean_accuracy", mode="max")

# Run Hyperband optimization
analysis = tune.run(
    train_model,
    config=search_space,
    num_samples=5,
    scheduler=scheduler
)

# Get the best hyperparameters
best_config = analysis.get_best_config(metric="mean_accuracy", mode="max")
print(f"Best hyperparameters: {best_config}")
