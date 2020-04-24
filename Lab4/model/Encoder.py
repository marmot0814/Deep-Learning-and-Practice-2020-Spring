import torch.nn as nn
import torch

class Encoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, device):
        super(Encoder, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size).to(device)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers).to(device)

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs).view(inputs.size(0), inputs.size(1), -1)
        outputs, hidden = self.lstm(embedded, hidden)
        return outputs, hidden

    def initHidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        )

