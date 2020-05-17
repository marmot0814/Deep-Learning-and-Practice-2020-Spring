import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config

class Generator(nn.Module):

    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.d = d
        self.conv1_1 = nn.Conv2d(d, d*2, 3, 1, 1)
        self.conv1_1_bn = nn.BatchNorm2d(d*2)
        self.conv1_2 = nn.Conv2d(24, d*2, 3, 1, 1)
        self.conv1_2_bn = nn.BatchNorm2d(d*2)
        self.conv2 = nn.Conv2d(d*4, d*2, 3, 1, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*2, 3, 1, 1)
        self.conv3_bn = nn.BatchNorm2d(d*2)
        self.conv4 = nn.Conv2d(d*2, d, 3, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d)
        self.conv5 = nn.Conv2d(d, d, 3, 1, 1)
        self.conv5_bn = nn.BatchNorm2d(d)
        self.conv5 = nn.Conv2d(d, d, 3, 1, 1)
        self.conv5_bn = nn.BatchNorm2d(d)
        self.conv6 = nn.Conv2d(d, 3, 3, 1, 1)
        self.optim = torch.optim.Adam(self.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
        self.criterion = nn.BCELoss()

    def forward(self, labels):
        noise = torch.rand((labels.size(0), self.d, 1, 1)).to(config.device)
        x = F.relu(self.conv1_1_bn(self.conv1_1(nn.Upsample(scale_factor=2)(noise))))
        y = F.relu(self.conv1_2_bn(self.conv1_2(nn.Upsample(scale_factor=2)(labels))))
        x = torch.cat([x, y], 1)
        x = F.relu(self.conv2_bn(self.conv2(nn.Upsample(scale_factor=2)(x))))
        x = F.relu(self.conv3_bn(self.conv3(nn.Upsample(scale_factor=2)(x))))
        x = F.relu(self.conv4_bn(self.conv4(nn.Upsample(scale_factor=2)(x))))
        x = F.relu(self.conv5_bn(self.conv5(nn.Upsample(scale_factor=2)(x))))
        x = torch.tanh(self.conv6(nn.Upsample(scale_factor=2)(x)))
        return x

    def Train(self, images, labels, D):
        (batch_size, n_class, _, _), image_sz = labels.size(), images.size(2)
        zeros = torch.zeros(batch_size).to(config.device)
        g_loss = -self.criterion(D(self.forward(labels), labels.expand(batch_size, n_class, image_sz, image_sz)), zeros)
        self.optim.zero_grad()
        g_loss.backward(retain_graph=True)
        self.optim.step()
        return g_loss.item()

    def Eval(self, labels):
#self.eval()
        return self.forward(labels)
