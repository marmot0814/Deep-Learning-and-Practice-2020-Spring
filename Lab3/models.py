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
import pyprind


class ResNet(nn.Module):
    def __init__(self, num, pretrained=False, lr=1e-4):
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
        self.classify = nn.Linear(pretrained_model._modules['fc'].in_features, 5)

        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum = 0.9, weight_decay = 5e-4)

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

    def Train(self, dataloader, criterion, device):
        self.train()
        loss, correct = 0, 0
        bar = pyprind.ProgPercent(len(dataloader), title="{} - Training...: ".format(self.name()))
        for idx, data in enumerate(dataloader):
            x, y = data
            inputs = x.to(device).float()
            labels = y.to(device).long().view(-1)

            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            
            t = criterion(outputs, labels)
            t.backward()

            loss += t
            self.optimizer.step()
            correct += (
                torch.max(outputs, 1)[1] == labels.long().view(-1)
            ).sum().item()

            bar.update(1)

        return loss.item(), correct * 100 / len(dataloader.dataset)

    def Test(self, dataloader, criterion, device):
        self.eval()
        with torch.no_grad():
            loss, correct = 0, 0
            bar = pyprind.ProgPercent(len(dataloader), title=f"{self.name()} - Testing...")
            for idx, data in enumerate(dataloader):
                x, y = data
                inputs = x.to(device).float()
                labels = y.to(device).long().view(-1)

                outputs = self.forward(inputs)
                loss += criterion(outputs, labels)

                correct += (
                    torch.max(outputs, 1)[1] == labels.long().view(-1)
                ).sum().item()
                bar.update(1)
            return loss.item(), correct * 100 / len(dataloader.dataset)

    def Predict(self, dataloader, device):
        self.eval()
        with torch.no_grad():
            bar = pyprind.ProgPercent(len(dataloader), title=f"{self.name()} - Testing...")
            result = []
            for idx, data in enumerate(dataloader):
                x, y = data
                inputs = x.to(device).float()
                labels = y.to(device).long().view(-1)

                outputs = self.forward(inputs)

                result += [x.item() for x in torch.max(outputs, 1)[1]]

                bar.update(1)
            return result

def ResNet18(device, pretrained=False):
    return ResNet(18, pretrained).to(device)

def ResNet50(device, pretrained=False):
    return ResNet(50, pretrained).to(device)
