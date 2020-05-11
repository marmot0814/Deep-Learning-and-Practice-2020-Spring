import torch.nn as nn
import time
import random
import json

from torch import optim

from model.Seq2Seq import Seq2Seq
from model.Encoder import EncoderRNN
from model.Decoder import DecoderRNN

from dataset.dataloader import TrainDataset, TestDataset

from config.config import device, hidden_size, MAX_LENGTH, cond_size, latent_size
from utils.func import timeSince, compute_bleu, KLD_weight, KL_loss, Teacher_forcing_ratio


def train(model, train_data, test_data, epoch_size, lr, criterion, print_every=1000):
    start = time.time()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epoch_size):
        
        ce, kl = model.Train(optimizer, train_dataset, criterion, 0.6, epoch)
        
        bleu = model.Test(test_dataset)
        with open('weight/record.json', 'r') as f:
            record = json.load(f)

        if not record.__contains__(model.name()) or record[model.name()] <= bleu:
            model.save()
            record[model.name()] = bleu
            with open('weight/record.json', 'w') as f:
                f.write(json.dumps(record, indent=2))
        print (f'epoch: {epoch} cross entropy: {ce} kl loss: {kl} bleu score: {bleu:}')
        
train_dataset = TrainDataset('dataset/official/train.txt')
test_dataset = TestDataset('dataset/official/test.txt')

model = Seq2Seq(hidden_size, len(train_dataset.dict.char2idx), cond_size, len(train_dataset.tense), latent_size).to(device).load()
train(model, train_dataset, test_dataset, 1000, 0.005, nn.CrossEntropyLoss())
