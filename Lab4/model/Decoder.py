import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, num_layers, device):
        super(Decoder, self).__init__()
        
        self.device = device
        self.embedding = nn.Embedding(output_size, hidden_size).to(self.device)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers).to(self.device)
        self.out = nn.Linear(hidden_size, output_size).to(self.device)
        self.softmax = nn.LogSoftmax(dim=1).to(self.device)

    def forward(self, input, hidden):
        batch_size = input.size(1)
        output = self.embedding(input).view(1, batch_size, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initInputs(self, batch_size):
        return torch.zeros(1, batch_size, device=self.device, dtype=torch.long)
