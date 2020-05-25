import torch
import torch.nn as nn

from models.gen.GenBlock import GenBlock

class GenNet(nn.Module):

    def __init__(self):
        super(GenNet, self).__init__()

        self.c_proj = nn.Linear(256, 256)

        self.l0 = nn.Linear(100 + 256, 1024 * 4 * 4)

        self.block1 = GenBlock(1024, 512)
        self.block2 = GenBlock(512 + 512, 256)
        self.block3 = GenBlock(256, 128)
        self.block4 = GenBlock(128, 64)

        self.bn = nn.BatchNorm2d(64)
        self.ac = nn.LeakyReLU()

        self.conv = nn.Conv2d(64, 3, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, z, y, fm):

        batch_size = y.size(0)

        cond_y = self.c_proj(y)
        x = torch.cat([z, cond_y], dim=1)

        x = self.l0(x)
        x = x.view(-1, 1024, 4, 4)
        x = self.block1(x, y)

        x = torch.cat([x, fm], dim=1)

        x = self.block2(x, y)
        x = self.block3(x, y)
        x = self.block4(x, y)

        x = self.bn(x)
        x = self.ac(x)
        x = self.conv(x)
        x = self.tanh(x)

        return x
