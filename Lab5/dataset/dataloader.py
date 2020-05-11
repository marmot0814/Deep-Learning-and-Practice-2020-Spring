import json
import torch
from torch.utils.data import Dataset
import numpy as np

from config.config import device


class Dictionary:

    def __init__(self, sigma):
        self.char2idx = {'SOS': 0, 'EOS': 1, 'UNK': 2}
        self.idx2char = {0: '', 1: '', 2: '?'}

        for c in sigma:
            if c in self.char2idx:
                continue
            idx = len(self.char2idx)
            self.char2idx[c] = idx
            self.idx2char[idx] = c

    def encode(self, w):
        return torch.tensor(
              [ self.char2idx[c] if c in self.char2idx else 2 for c in w ]
            + [ 1 ],
        device=device).view(-1, 1)

    def decode(self, t):
        return ''.join([self.idx2char[v.item()] for v in t.view(-1)])


class TrainDataset(Dataset):
    
    def __init__(self, path):
        self.data = np.loadtxt(path, dtype=np.str).reshape(-1)
        self.dict = Dictionary("abcdefghijklmnopqrstuvwxyz")
        self.tense = {
            'sp': 0, 
            'tp': 1, 
            'pg': 2, 
            'p': 3
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], index % len(self.tense)
    
class TestDataset(Dataset):

    def __init__(self, path):
        self.data = np.loadtxt(path, dtype=np.str)
        self.dict = Dictionary("abcdefghijklmnopqrstuvwxyz")
        self.tense = {
            'sp': 0, 
            'tp': 1, 
            'pg': 2, 
            'p': 3
        }
        self.target = [
            ['sp',  'p'],
            ['sp',  'pg'],
            ['sp',  'tp'],
            ['sp',  'tp'],
            ['p',   'tp'],
            ['sp',  'pg'],
            ['p',   'sp'],
            ['pg', 'sp'],
            ['pg', 'p'],
            ['pg', 'tp']
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.tense[self.target[index][0]], self.data[index][1], self.tense[self.target[index][1]]
