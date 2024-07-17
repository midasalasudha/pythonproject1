# k-NN with PyTorch

import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()


# Implementing the k-NN Algorithm

def euclidean_distance(a, b):
    return torch.sqrt(torch.sum((a - b) ** 2, dim=1))


def knn_predict(X_train, y_train, X_test, k):
    y_pred = []
    for test_point in X_test:
        distances = euclidean_distance(X_train, test_point)
        _, indices = torch.topk(distances, k, largest=False)
        nearest_labels = y_train[indices]
        majority_label = torch.mode(nearest_labels).values.item()
        y_pred.append(majority_label)
    return torch.tensor(y_pred)


# Predict using k-NN with k=3
k = 3
y_pred = knn_predict(X_train_tensor, y_train_tensor, X_test_tensor, k)

# Evaluating the Model

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Generate classification report
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print('Classification Report:')
print(report)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()




