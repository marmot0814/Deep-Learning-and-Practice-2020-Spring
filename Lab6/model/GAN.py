import torch
import torch.nn as nn
from torchvision.utils import save_image

from model.Generator import Generator
from model.Discriminator import Discriminator

from config import config

import random

class GAN(nn.Module):

    def __init__(self, d=128):
        super(GAN, self).__init__()
        self.d = d
        self.D = Discriminator(d)
        self.G = Generator(d)
        self.c_embedding = nn.Embedding(9, 8).to(config.device)
        self.s_embedding = nn.Embedding(4, 8).to(config.device)

    def forward(self, input):
        pass

    def Train(self, images, labels):

        cnt = config.n_critic
        while cnt > 0:
            d_loss = self.D.Train(images, labels, self.G, self.c_embedding, self.s_embedding)
            cnt -= 1

        self.eval()
        print ("After D step")
        print ("real image", self.D.Eval(images, labels, self.c_embedding, self.s_embedding).mean().item())
        print ("fake image", self.D.Eval(self.G.Eval(labels, self.c_embedding, self.s_embedding), labels, self.c_embedding, self.s_embedding).mean().item())

        cnt = 1
        while cnt > 0:
            g_loss = self.G.Train(images, labels, self.D, self.c_embedding, self.s_embedding)
            cnt -= 1
        
        self.eval()
        print ("After G step")
        print ("real image", self.D.Eval(images, labels, self.c_embedding, self.s_embedding).mean().item())
        print ("fake image", self.D.Eval(self.G.Eval(labels, self.c_embedding, self.s_embedding), labels, self.c_embedding, self.s_embedding).mean().item())
        print (f'D_loss: {d_loss}, G_loss: {g_loss}')

    def Eval(self, labels):
        return self.G.Eval(labels, self.c_embedding, self.s_embedding)

    def save(self, epoch):
        torch.save(self.state_dict(), 'weight/' + self.name() + '/' + self.name() + f'-Epoch-{epoch}-0')

    def load(self, epoch):
        self.load_state_dict(torch.load('weight/' + self.name() + f'/{self.name()}-Epoch-{epoch}-0'))
        return self

    def name(self):
        return f"GAN-{self.d}"
