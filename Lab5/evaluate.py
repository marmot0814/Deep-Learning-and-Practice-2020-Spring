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

words = []
torch.random.manual_seed(460)
for i in range(100):
    words.append(model.sample(train_dataset))
print (f'Gaussian score: {Gaussian_score(words)}')
