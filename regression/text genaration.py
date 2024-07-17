import torch
import torch.nn as nn
import torch.optim as optim

# Sample text data
text = "hello world this is a simple text generation example "
chars = sorted(list(set(text)))
print('Sorted List', chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
print(f'Char:Idx  {char_to_idx}  Idx:Char {idx_to_char}  ')

# Hyperparameters
input_size = len(chars)
hidden_size = 128
output_size = len(chars)
n_layers = 1
seq_length = 10
learning_rate = 0.001
num_epochs = 500


# RNN model definition
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out.reshape(out.size(0) * out.size(1), out.size(2)))
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)


# Prepare the data
def char_tensor(string):
    tensor = torch.zeros(len(string), len(chars))
    for c in range(len(string)):
        tensor[c][char_to_idx[string[c]]] = 1
    return tensor


# Training the model
model = SimpleRNN(input_size, hidden_size, output_size, n_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(num_epochs):
    hidden = model.init_hidden(1)
    for i in range(0, len(text) - seq_length, seq_length):
        seq_in = text[i:i + seq_length]
        seq_out = text[i + 1:i + seq_length + 1]

        seq_in_tensor = char_tensor(seq_in).unsqueeze(0)
        seq_out_tensor = torch.tensor([char_to_idx[ch] for ch in seq_out])

        optimizer.zero_grad()
        output, hidden = model(seq_in_tensor, hidden)
        loss = criterion(output, seq_out_tensor)
        loss.backward(retain_graph=True)  # Use retain_graph=True to keep the graph for next iteration
        optimizer.step()

        hidden = hidden.detach()  # Detach hidden state to prevent backpropagating through the entire history

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# Text generation
def generate_text(model, start_text, length):
    model.eval()
    hidden = model.init_hidden(1)
    input_seq = char_tensor(start_text).unsqueeze(0)

    generated_text = start_text
    for _ in range(length):
        output, hidden = model(input_seq, hidden)
        _, top_idx = output.topk(1)
        predicted_char = idx_to_char[top_idx[-1].item()]
        generated_text += predicted_char
        input_seq = char_tensor(predicted_char).unsqueeze(0)

    return generated_text


# Generate new text
start_text = "hello"
generated_text = generate_text(model, start_text, 100)
print("Generated text:")
print(generated_text)

