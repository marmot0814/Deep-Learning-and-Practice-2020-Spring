import torch.nn as nn
import torch.nn.functional as F

from config.config import device

class DecoderRNN(nn.Module):

    def __init__(self, num_of_word, hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_of_word, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, num_of_word)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def name(self):
        return f'DecoderRNN-{self.hidden_size}'
