import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_

from models.dis.DisBlock import DisBlock


class DisNet(nn.Module):

    def __init__(self):
        super(DisNet, self).__init__()

        self.ac = nn.LeakyReLU()

        self.block1 = DisBlock(3, 64, True)
        self.block2 = DisBlock(64, 128)
        self.block3 = DisBlock(128, 256)
        self.block4 = DisBlock(256, 512)
        self.block5 = DisBlock(512, 1024)

        self.l6 = spectral_norm(nn.Linear(1024, 1))
        
        self.condition_projector = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)
        )
        
        self.aux_objective = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 24)
        )
        
    def forward(self, x, c, prev_image):
        prev_image = self.block1(prev_image)
        prev_image = self.block2(prev_image)
        prev_image = self.block3(prev_image)

        y = self.condition_projector(c)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x - prev_image

        x = self.block4(x)
        x = self.block5(x)

        x = self.ac(x)
        x = x.sum(dim=(2, 3))

        aux = torch.sigmoid(self.aux_objective(x))
        out = self.l6(x).view(-1)

        c = (y * x).sum(dim = 1)
        out = out + c

        return out, aux
