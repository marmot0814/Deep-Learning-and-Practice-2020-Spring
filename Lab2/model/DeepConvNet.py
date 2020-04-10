import torch
import torch.nn as nn
import torch.optim as optim

class DeepConvNet(nn.Module):

    def __init__(self, acti=nn.ReLU, dropout=0.25, optim=optim.Adam, lr=0.001):
        super(DeepConvNet, self).__init__()

        self.acti = acti
        self.dropout = dropout

        self.conv0 = nn.Sequential(
            nn.Conv2d( 1, 25, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.Conv2d(25, 25, kernel_size=(2, 1), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(25),
            acti(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(50),
            acti(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(100),
            acti(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(200),
            acti(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout)
        )

        self.classify = nn.Sequential(
            nn.Linear(8600, 2, bias=True),
        )

        self.optimizer = optim(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.classify[0].in_features)
        x = self.classify(x)
        return x

    def name(self):
        return f'DeepConvNet-{self.acti().__str__().split("(")[0]}-{self.dropout}'
