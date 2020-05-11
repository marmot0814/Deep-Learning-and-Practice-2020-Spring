import torch
import torch.nn as nn

from config.config import device

class EncoderRNN(nn.Module):

    def __init__(self, num_of_word, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_of_word, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(input.size(0), 1, -1)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def name(self):
        return f'EncoderRNN-{self.hidden_size}'
