import torch.nn as nn
import time
import random
import json

from torch import optim

from model.Seq2Seq import Seq2Seq
from model.Encoder import EncoderRNN
from model.Decoder import DecoderRNN

from dataset.dataloader import DataLoader

from config.config import device, hidden_size, teacher_forcing_ratio, MAX_LENGTH
from utils.func import timeSince, compute_bleu


def train(model, dataloader, n_iters, lr, criterion, print_every=100):
    start = time.time()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    print_loss_total = 0
    for iter in range(1, n_iters + 1):
        optimizer.zero_grad()
        pair = random.choice(dataloader.train_data)
        input = dataloader.encode(pair[0])
        target = dataloader.encode(pair[1])

        loss = 0
        outputs = model(input, target, random.random() < teacher_forcing_ratio)
        outputs = outputs.narrow(0, 0, min(target.size(0), outputs.size(0)))
        for idx, output in enumerate(outputs):
            loss += criterion(outputs[idx], target[idx])

        loss.backward()
        optimizer.step()

        print_loss_total += loss.item() / target.size(0)
        output = model.predict(input, dataloader)

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print ('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

            avg_bleu = model.test(dataloader)

            with open('weight/record.json') as f:
                record = json.load(f)
        
            if not record.__contains__(model.name()) or record[model.name()] < avg_bleu:
                model.save()
                record[model.name()] = avg_bleu
                with open('weight/record.json', 'w') as f:
                    f.write(json.dumps(record, indent=2, sort_keys=True))

            print (f'Avg Bleu: {avg_bleu:.2f}%')
            print ('')

dataloader = DataLoader('dataset/official_test/')

model = Seq2Seq(
            EncoderRNN(len(dataloader.dict.char2idx), hidden_size).to(device),
            DecoderRNN(hidden_size, len(dataloader.dict.char2idx)).to(device)
        )

train(model, dataloader, 1000000, 0.01, nn.CrossEntropyLoss(), 1000)
