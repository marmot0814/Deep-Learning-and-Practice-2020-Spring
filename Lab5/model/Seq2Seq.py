import torch
import torch.nn as nn
import random

from utils.func import compute_bleu, KL_loss, KLD_weight
from model.Encoder import EncoderRNN
from model.Decoder import DecoderRNN

from config.config import device, MAX_LENGTH

class Seq2Seq(nn.Module):

    def __init__(self, hidden_size, num_of_word, cond_size, num_of_cond, latent_size):
        super(Seq2Seq, self).__init__()

        self.cond_embedding = nn.Embedding(num_of_cond, cond_size)

        self.encoder = EncoderRNN(num_of_word, hidden_size)

        self.mean = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)
        self.latent2hidden = nn.Linear(latent_size + cond_size, hidden_size)
        self.latent2cell = nn.Linear(latent_size + cond_size, hidden_size)

        self.decoder = DecoderRNN(num_of_word, hidden_size)

        self.hidden_size = hidden_size
        self.cond_size = cond_size
        self.latent_size = latent_size

    def condition(self, c):
        return self.cond_embedding(torch.LongTensor([c]).to(device)).view(1, 1, -1)

    def sample_z(self):
        return torch.normal(
            torch.FloatTensor([0] * (self.latent_size)),
            torch.FloatTensor([1] * (self.latent_size))
        ).to(device).view(1, 1, -1)

    def initHidden(self, hidden, cell, cond):
        return (
            torch.cat((hidden, cond), dim=2),
            cell
        )

    def forward(self, input, input_c, target, target_c, use_teacher_forcing):
        output, hidden = self.encoder(
            input,
            self.initHidden(
                torch.zeros(self.hidden_size - self.cond_size, device=device).view(1, 1, -1),
                torch.zeros(self.hidden_size, device=device).view(1, 1, -1),
                self.condition(input_c)
            )
        )
        m = self.mean(hidden[0])
        lgv = self.logvar(hidden[0])
        z = self.sample_z() * torch.exp(lgv / 2) + m

        hidden = (
            self.latent2hidden(torch.cat((z, self.condition(target_c)), dim=2).reshape(-1)).view(1, 1, -1),
            self.latent2cell(torch.cat((z, self.condition(target_c)), dim=2).reshape(-1)).view(1, 1, -1)
        )

        input = torch.tensor([[0]], device=device)
        ret = []
        for i in range(MAX_LENGTH):
            output, hidden = self.decoder(input, hidden)
            if use_teacher_forcing:
                if i == target.size(0):
                    break
                ret.append(output)
                input = target[i]
            else:
                ret.append(output)
                topv, topi = output.topk(1)
                input = topi.squeeze().detach()
                if input.item() == 1:
                    break
        return torch.stack(ret), m, lgv

    def name(self):
        return self.encoder.name() + '-' + self.decoder.name()

    def load(self):
        self.load_state_dict(torch.load('weight/' + self.name()))
        return self

    def save(self):
        torch.save(self.state_dict(), 'weight/' + self.name()) 

    def Train(self, optimizer, train_dataset, criterion, teacher_forcing_ratio, epoch):
        total_ce, total_kl = 0, 0
        idx_pool = list(range(len(train_dataset)))
        random.shuffle(idx_pool)
        cnt = 0
        for idx in idx_pool:
            optimizer.zero_grad()
            data, c = train_dataset[idx]
            t = train_dataset.dict.encode(data)
            o, m, lgv = self.forward(t, c, t, c, random.random() < teacher_forcing_ratio)
            o = o.narrow(0, 0, min(t.size(0), o.size(0)))

            ce = 0
            for (_o, _t) in zip(o, t):
                ce += criterion(_o, _t)
            ce /= len(o)

            kl_loss = KL_loss(m, lgv)
            (ce + KL_loss(m, lgv) * KLD_weight(epoch * len(train_dataset) + cnt)).backward()

            total_ce += ce.item()
            total_kl += kl_loss.item()
            optimizer.step()
            cnt += 1
        return total_ce / len(train_dataset), total_kl / len(train_dataset)
    
    def Test(self, test_dataset, display=False):
        total_bleu = 0
        for idx in range(len(test_dataset)):
            _i, i_c, t, t_c = test_dataset[idx]
            i = test_dataset.dict.encode(_i)
            o, m, lgv = self.forward(i, i_c, None, t_c, False)
            o = test_dataset.dict.decode(o.argmax(dim=2).view(-1, 1))
            total_bleu += compute_bleu(o, t)
            if not display:
                continue
            print (f"input: {_i:20}, target: {t:20}, output: {o:20}")
        return total_bleu / len(test_dataset)

    def predict(self, i, i_c, t_c, dataset):
        i = dataset.dict.encode(i)
        o, m, lgv = self.forward(i, i_c, None, t_c, False)
        o = dataset.dict.decode(o.argmax(dim=2).view(-1, 1))
        return o

    def sample(self, dataset):
        z = self.sample_z()
        word = []
        for c in range(4):
            hidden = (
                self.latent2hidden(torch.cat((z, self.condition(c)), dim=2).reshape(-1)).view(1, 1, -1),
                self.latent2cell(torch.cat((z, self.condition(c)), dim=2).reshape(-1)).view(1, 1, -1)
            )

            input = torch.tensor([[0]], device=device)
            ret = []
            for i in range(MAX_LENGTH):
                output, hidden = self.decoder(input, hidden)
                ret.append(output)
                topv, topi = output.topk(1)
                input = topi.squeeze().detach()
                if input.item() == 1:
                    break

            o = torch.stack(ret)
            o = dataset.dict.decode(o.argmax(dim=2).view(-1, 1))
            word.append(o)
        return word
