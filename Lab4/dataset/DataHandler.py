import json
import torch
import random
from config import config

class Dictionary:

    def __init__(self, path):
        self.path = path

        self.char2idx = {'SOS': 0, 'EOS': 1, 'PAD': 2, 'UNK': 3}
        self.idx2char = {0: 'SOS', 1: 'EOS', 2: 'PAD', 3: 'UNK'}
        self.num_chars = 4
        self.pairs = ([], [])

        self.__load_data()
        self.__analysis()
        self.__construct()

    def __load_data(self):
        with open(self.path) as f:
            self.raw_data = json.load(f)[:2]

    def __analysis(self):
        for item in self.raw_data:
            self.__analysis_word(item['target'])
            for word in item['input']:
                self.__analysis_word(word)

    def __analysis_word(self, w):
        for c in w:
            if c not in self.char2idx:
                idx = self.num_chars
                self.char2idx[c] = idx
                self.idx2char[idx] = c
                self.num_chars += 1

    def __construct(self):
        for item in self.raw_data:
            for word in item['input']:
                self.pairs[0].append(word)
                self.pairs[1].append(item['target'])

    def encode(self, ws):
        l = len(max(ws, key=len)) + 1
        return torch.stack([
            torch.tensor(
                  [ self.char2idx[c] for c in w ]
                + [ self.char2idx['EOS'] ]
                + [ self.char2idx['PAD'] for x in range(l - len(w) - 1) ]
            , device=config.device).view(-1, 1) for w in ws
        ], dim=1).view(l, -1)

    def decode(self, ts):
        return [
            ''.join([
                self.idx2char[v.item()] for v in t if v.item() > 3
            ]) 
        for t in ts.T]


class DataHandler:

    def __init__(self, path):
        self.dict = Dictionary(path)
        self.vocab_size = len(self.dict.char2idx)

    def __random_shuffle(self, pairs):
        c = list(zip(pairs[0], pairs[1]))
        random.shuffle(c)
        a, b = zip(*c)
        return (list(a), list(b))

    def encode(self, ws):
        return self.dict.encode(ws)

    def decode(self, ts):
        return self.dict.decode(ts)

    def mini_batches(self, batch_size = 32):
        self.dict.pairs = self.__random_shuffle(self.dict.pairs)
        for idx in range(0, len(self.dict.pairs[0]), batch_size):
            I = self.dict.pairs[0][ idx : idx + batch_size ]
            O = self.dict.pairs[1][ idx : idx + batch_size ]
            yield I, O
