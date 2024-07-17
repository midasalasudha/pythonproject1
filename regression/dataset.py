import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# Sample dataset
class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# Define the RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Get the output from the last time step
        return out


# Hyperparameters
input_size = 10  # Number of features in the input
hidden_size = 20
output_size = 2  # Number of output classes
num_layers = 2
num_epochs = 10
learning_rate = 0.001

# Generate some dummy data
sequences = torch.randn(100, 5, input_size)  # 100 sequences of length 5
targets = torch.randint(0, 2, (100,))  # Binary classification

# Create dataset and dataloader
dataset = SequenceDataset(sequences, targets)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Instantiate the model, loss function, and optimizer
model = SimpleRNN(input_size, hidden_size, output_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete")

# Save the model
torch.save(model.state_dict(), 'simple_rnn_model.pth')

# Load the model
modele = SimpleRNN(input_size, hidden_size, output_size, num_layers)
modele.load_state_dict(torch.load('simple_rnn_model.pth'))
modele.eval()

# Make a prediction
with torch.no_grad():
    sample_input = torch.randn(1, 5, input_size)  # Single sequence of length 5
    print(f' SI :  {sample_input}')
    prediction = modele(sample_input)
    print(f' Pred : {prediction}')
    predicted_class = torch.argmax(prediction, dim=1)
    print(f'Predicted class: {predicted_class.item()}')

'''
Training Loop:
For each epoch, the model processes the training data in batches.
The loss is calculated using the criterion (CrossEntropyLoss).
The optimizer updates the model parameters to minimize the loss.
After each epoch, the average loss for that epoch is printed.

Model Saving:
After training, the model's state dictionary is saved to a file named simple_rnn_model.pth.

Model Loading:
The saved model state dictionary is loaded back into the model.

Prediction:
A random sample input sequence is generated.
The model makes a prediction for this input.
The predicted class is printed.
'''

