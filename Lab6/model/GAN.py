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

    def Train(self, images, labels, counter):
        (batch_size, n_class), image_sz = labels.size(), images.size(2)
        labels = labels.view(batch_size, n_class, 1, 1)

        acc, cnt = 1, 5
        while cnt >= 0:

            d_loss = self.D.Train(images, labels, self.G)
#            acc = self.D.Eval(self.G.Eval(labels), labels).mean().item()
            cnt -= 1

        print ("After D step")
        print ("real image", self.D.Eval(images, labels).mean().item())
        print ("fake image", self.D.Eval(self.G.Eval(labels), labels).mean().item())

        acc, cnt = 0, 1
        while cnt >= 0:

            g_loss = self.G.Train(images, labels, self.D)
#            acc = self.D.Eval(self.G.Eval(labels), labels).mean().item()
            cnt -= 1
        
        print ("After G step")
        print ("real image", self.D.Eval(images, labels).mean().item())
        print ("fake image", self.D.Eval(self.G.Eval(labels), labels).mean().item())

        print (f'D_loss: {d_loss}, G_loss: {g_loss}')
        '''
        target_image = (self.G.Eval(labels)[random.randint(0, labels.size(0) - 1)] + 1) / 2
        save_image(target_image, f"Hello_{counter}.png")
        '''


    def Eval(self, labels):
        batch_size, n_class = labels.size()
        labels = labels.view(batch_size, n_class, 1, 1).to(config.device)
        return self.G.Eval(labels)

    def save(self, epoch):
        torch.save(self.state_dict(), 'weight/' + self.name() + '/' + self.name() + f'-Epoch-{epoch}')

    def name(self):
        return f"GAN-{self.d}"
