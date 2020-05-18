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
        self.criterion = nn.BCELoss()

    def forward(self, input):
        pass

    def Train(self, images, labels):
        (batch_size, n_class), image_sz = labels.size(), images.size(2)
        labels = labels.view(batch_size, n_class, 1, 1)

        self.D.train()
        self.G.train()
        acc, cnt = 1, 5
        while cnt >= 0:
            d_loss = self.D.Train(images, labels, self.G)
            cnt -= 1

        self.D.eval()
        self.G.eval()
        print ("After D step")
        print ("real image", self.D.Eval(images, labels).mean().item())
        print ("fake image", self.D.Eval(self.G.Eval(labels), labels).mean().item())

        self.D.train()
        self.G.train()
        acc, cnt = 0, 1
        while cnt >= 0:
            g_loss = self.G.Train(images, labels, self.D)
            cnt -= 1
        
        self.D.eval()
        self.G.eval()
        print ("After G step")
        print ("real image", self.D.Eval(images, labels).mean().item())
        print ("fake image", self.D.Eval(self.G.Eval(labels), labels).mean().item())
        print (f'D_loss: {d_loss}, G_loss: {g_loss}')

    def Eval(self, labels):
        batch_size, n_class = labels.size()
        labels = labels.view(batch_size, n_class, 1, 1).to(config.device)
        return self.G.Eval(labels)

    def save(self, epoch):
        torch.save(self.state_dict(), 'weight/' + self.name() + '/' + self.name() + f'-Epoch-{epoch}-0')

    def load(self, epoch):
        self.load_state_dict(torch.load('weight/' + self.name() + f'/{self.name()}-Epoch-{epoch}-0'))
        return self


    def name(self):
        return f"GAN-{self.d}"
