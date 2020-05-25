import torch.nn as nn

from torch.nn.utils import spectral_norm

class DisBlock(nn.Module):

    def __init__(self, ic, oc, first = False):
        super(DisBlock, self).__init__()
        self.c0 = spectral_norm(nn.Conv2d(ic, oc, 1, 1, 0))
        self.c1 = spectral_norm(nn.Conv2d(ic, oc, 3, 1, 1))
        self.c2 = spectral_norm(nn.Conv2d(oc, oc, 3, 1, 1))
        self.ac = nn.LeakyReLU()
        self.downsample = nn.AvgPool2d(2)
        self.first = first

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

    def residual(self, x):
        if not self.first:
            x = self.ac(x)
        x = self.ac(self.c1(x))
        x = self.downsample(self.c2(x))
        return x

    def shortcut(self, x):
        if self.first:
            return self.c0(self.downsample(x))
        else:
            return self.downsample(self.c0(x))
