import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.cuda.comm as comm
from torch.autograd.function import once_differentiable
import time
import functools
from torch import optim


class ResNet(nn.Module):
    def __init__(self, num, fc_in, pretrained=False):
        super(ResNet, self).__init__()

        self.pretrained = pretrained
        self.num = num
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(num)](pretrained=pretrained)
        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classify = nn.Linear(fc_in, 5)

        self.optimizer = optim.SGD(self.parameters(), lr=1e-4, momentum = 0.9, weight_decay = 5e-4)


        del pretrained_model

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.classify(x)

        return x

    def name(self):
        return f"ResNet{self.num}" + ("-pretrained" if self.pretrained else "")

    def load(self):
        self.load_state_dict(torch.load('weight/' + self.name()))
        return self

def ResNet18(device, pretrained=False):
    return ResNet(18, 512, pretrained).to(device)

def ResNet50(device, pretrained=False):
    return ResNet(50, 2048, pretrained).to(device)
