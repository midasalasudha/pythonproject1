

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import os


class ButterflyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        # Create a label to index mapping
        # self.label_to_index = {label: idx for idx, label in enumerate(self.annotations.iloc[:, 1].unique())}
        if 'label' in self.annotations.columns:
            self.label_to_index = {label: idx for idx, label in enumerate(self.annotations['label'].unique())}
        else:
            self.label_to_index = {}  # Handle cases where labels are not provided

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.annotations.iloc[idx, 1]
        label = self.label_to_index[label]  # Convert label to index
        if self.transform:
            image = self.transform(image)
        return image, label


# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, len(trainset.label_to_index))  # Adjust output layer based on number of classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Training loop
def train_model(model, trainloader, criterion, optimizer, num_epochs=10):
    net.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}')
                running_loss = 0.0
    print('Finished Training')


def evaluate_model(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f}%')
    return accuracy


if __name__ == '__main__':
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to 128x128
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
    ])
    # Correct paths to CSV files and root directories
    trainset = ButterflyDataset(csv_file='./butterfly_dataset/butterfly_dataset/Training_set.csv',
                                root_dir='./butterfly_dataset/butterfly_dataset/train', transform=transform)
    testset = ButterflyDataset(csv_file='./butterfly_dataset/butterfly_dataset/Testing_set.csv',
                               root_dir='./butterfly_dataset/butterfly_dataset/test', transform=transform)
    # Create DataLoaders
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    # Initialize the model, criterion, and optimizer
    net = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    train_model(net, trainloader, criterion, optimizer, num_epochs=10)
    test_accuracy = evaluate_model(net, testloader)

