import torch
import torch.nn as nn
import torch.optim as optim
import torchtext; torchtext.disable_torchtext_deprecation_warning()

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import IMDB
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader

# Load and preprocess the dataset
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


# Text and label processing
def text_pipeline(x): return vocab(tokenizer(x))
def label_pipeline(x): return 1 if x == 'pos' else 0


# Collate function for DataLoader
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list, text_list, offsets


# Define the RNN model
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers=1):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_size, sparse=True)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        out, _ = self.rnn(embedded.unsqueeze(1))
        out = self.fc(out[:, -1, :])
        return out


#DRIVER
# Load IMDB dataset
train_iter, test_iter = IMDB(split=('train', 'test'))


# Build vocabulary
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Reload dataset with map style
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)


# DataLoader
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)


# Hyperparameters
vocab_size = len(vocab)
embed_size = 100
hidden_size = 256
output_size = 2  # binary classification
num_layers = 2

# Initialize model, loss function, and optimizer
model = SentimentRNN(vocab_size, embed_size, hidden_size, output_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for labels, text, offsets in train_dataloader:
        optimizer.zero_grad()
        output = model(text, offsets)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader):.4f}')

print('Test data :  ',test_dataloader)

# Testing the model
model.eval()
total_acc, total_count = 0, 0
with torch.no_grad():
    for labels, text, offsets in test_dataloader:
        output = model(text, offsets)
        pred = output.argmax(1)
        total_acc += (pred == labels).sum().item()
        total_count += labels.size(0)
        #print('Pred : ', pred)
accuracy = total_acc / total_count
print(f'Test Accuracy: {accuracy:.4f}')
