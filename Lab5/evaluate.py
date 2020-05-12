import torch

from dataset.dataloader import TrainDataset, TestDataset

from model.Seq2Seq import Seq2Seq
from model.Encoder import EncoderRNN
from model.Decoder import DecoderRNN

from config.config import hidden_size, device, cond_size, latent_size

from utils.func import compute_bleu, Gaussian_score

train_dataset = TrainDataset('dataset/official/train.txt')
test_dataset = TestDataset('dataset/official/test.txt')

model = Seq2Seq(hidden_size, len(train_dataset.dict.char2idx), cond_size, len(train_dataset.tense), latent_size).to(device).load()

torch.random.manual_seed(243)
print (f'BLEU-4 score: {model.Test(test_dataset, True)}')
print ('=' * 100)

S = set()
with open('dataset/official/train.txt', 'r') as f:
    for line in f.readlines():
        S.add(tuple(line.strip().split(' ')))

words = []
torch.random.manual_seed(460)
for i in range(100):
    words.append(model.sample(train_dataset))
    if tuple(words[-1]) in S:
        print (" ---> ", end='')
    else:
        print ("      ", end='')
    for w in words[-1]:
        print (f'{w:15}', end='')
    if tuple(words[-1]) in S:
        print (" <--- ")
    else:
        print ("      ")

print (f'Gaussian score: {Gaussian_score(words)}')
