import json
import torch

from config.config import device

class Dictionary:

    def __init__(self, words):
        self.char2idx = {'SOS': 0, 'EOS': 1, 'UNK': 2}
        self.idx2char = {0: '', 1: '', 2: '?'}

        self.__analysis(words)

    def __analysis(self, words):
        for word in words:
            self.__analysis_word(word)

    def __analysis_word(self, word):
        for char in word:
            if char in self.char2idx:
                continue
            idx = len(self.char2idx)
            self.char2idx[char] = idx
            self.idx2char[idx] = char
        
    def encode(self, w):
        return torch.tensor(
              [ self.char2idx[c] if c in self.char2idx else 2 for c in w ]
            + [ 1 ],
        device=device).view(-1, 1)

    def decode(self, t):
        return ''.join([self.idx2char[v.item()] for v in t.view(-1)])

class DataLoader:

    def __init__(self, path):
        self.train_data = self.__load(path + "train.json", True)
        self.test_data  = self.__load(path + "test.json", False)
        self.dict = Dictionary(sum([ [ p[0], p[1] ] for p in self.train_data], []))

    def __load(self, path, base):
        with open(path, 'r') as f:
            return [ 
                (input, item['target']) for item in json.load(f) 
                for input in item['input'] + [item['target']] * base
            ]

    def encode(self, w):
        return self.dict.encode(w)

    def decode(self, t):
        return self.dict.decode(t)
