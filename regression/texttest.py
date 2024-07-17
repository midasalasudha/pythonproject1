import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import GloVe


# Load pre-trained GloVe embeddings
glove = GloVe(name='6B', dim=100)

# Example: Get the vector for the word 'computer'
vector = glove['computer']
print('Vec for Computer :  ',vector)

# Integrating Embeddings into a Neural Network

import torch
import torch.nn as nn
import torch.optim as optim

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, pretrained_embeddings):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = False  # Freeze embeddings
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Example usage
vocab_size = len(glove.stoi)
embed_size = 100
hidden_size = 256
output_size = 2  # Example for binary classification

# Create an embedding matrix from the pre-trained embeddings
embedding_matrix = torch.zeros((vocab_size, embed_size))
for word, idx in glove.stoi.items():
    embedding_matrix[idx] = glove[word]

model = TextClassificationModel(vocab_size, embed_size, hidden_size, output_size, embedding_matrix)

# Dummy data and training loop for illustration
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
data = torch.randint(0, vocab_size, (10, 5))  # Example data
labels = torch.randint(0, 2, (10,))  # Example labels

for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/5], Loss: {loss.item():.4f}')


# Testing
model.eval()

# Dummy test data (replace with actual test data)
test_data = ["example sentence for testing", "another test sentence"]

# Preprocess and tokenize the test data
def preprocess_sentence(sentence, glove):
    tokens = sentence.split()
    indices = [glove.stoi.get(token, glove.stoi['unk']) for token in tokens]  # Use 'unk' for unknown words
    return torch.tensor(indices)

test_tensors = [preprocess_sentence(sentence, glove) for sentence in test_data]

# Pad the test sequences to the same length (assuming a max length)
max_length = max(len(tensor) for tensor in test_tensors)
padded_tensors = [torch.cat([tensor, torch.zeros(max_length - len(tensor))]) for tensor in test_tensors]
test_tensors = torch.stack(padded_tensors).long()

# Generate predictions
with torch.no_grad():
    outputs = model(test_tensors)
    _, predicted = torch.max(outputs, 1)

# Assuming the labels are 0 and 1 for binary classification
label_names = ["Class 0", "Class 1"]

# Print the predictions
for i, sentence in enumerate(test_data):
    print(f"Sentence: {sentence}")
    print(f"Predicted Label: {label_names[predicted[i].item()]}")


