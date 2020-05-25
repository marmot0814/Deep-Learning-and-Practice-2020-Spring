import torch.nn as nn

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, ic, d):
        super().__init__()
        self.ic = ic
        self.bn = nn.BatchNorm2d(ic, affine=False)
        self.fc = nn.Linear(d, ic * 2)

        self.fc.weight.data[:, :ic] = 1
        self.fc.weight.data[:, ic:] = 0

    def forward(self, x, c):
        c = self.fc(c)
        g = c[:, :self.ic].view(-1, self.ic, 1, 1)
        b = c[:, self.ic:].view(-1, self.ic, 1, 1)
        x = self.bn(x)
        return x.mul(g).add(b)
