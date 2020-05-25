import torch.nn as nn

from models.ConditionalBatchNorm import ConditionalBatchNorm2d

class GenBlock(nn.Module):
    def __init__(self, ic, oc):
        super(GenBlock, self).__init__()

        self.ac = nn.LeakyReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.cbn1 = ConditionalBatchNorm2d(ic, 256)
        self.cbn2 = ConditionalBatchNorm2d(oc, 256)

        self.c0 = nn.Conv2d(ic, oc, 1, 1, 0)
        self.c1 = nn.Conv2d(ic, oc, 3, 1, 1)
        self.c2 = nn.Conv2d(oc, oc, 3, 1, 1)

    def forward(self, x, y):
        return self.shortcut(x) + self.residual(x, y)

    def residual(self, x, y):
        x = self.ac(self.cbn1(x, y))
        x = self.c1(self.upsample(x))
        x = self.ac(self.cbn2(x, y))
        return self.c2(x)
        
    def shortcut(self, x):
        return self.c0(self.upsample(x))
