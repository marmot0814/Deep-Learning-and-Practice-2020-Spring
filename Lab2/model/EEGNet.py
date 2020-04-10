import torch
import torch.nn as nn
import torch.optim as optim

class EEGNet(nn.Module):

    def __init__(self, acti=nn.ReLU, dropout=0.25, optim=optim.Adam, lr=0.01):
        super(EEGNet, self).__init__()

        self.acti = acti
        self.dropout = dropout

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            acti(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=dropout)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32),
            acti(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=dropout)
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )
    
        self.optimizer = optim(self.parameters(), lr=lr)
        

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(-1, self.classify[0].in_features)
        x = self.classify(x)
        return x

    def Test(self, dataloader, criterion, device):
        self.eval()
        with torch.no_grad():
            loss, correct = 0, 0
            for idx, data in enumerate(dataloader):
                x, y = data
                inputs = x.to(device)
                labels = y.to(device).long().view(-1)

                outputs = self.forward(inputs)
                loss += criterion(outputs, labels)

                correct += (
                    torch.max(outputs, 1)[1] == labels.long().view(-1)
                ).sum().item()
            return loss.item(), correct * 100 / len(dataloader.dataset)

    def Train(self, dataloader, criterion, device):
        self.train()
        loss, correct = 0, 0
        for idx, data in enumerate(dataloader):
            x, y = data
            inputs = x.to(device)
            labels = y.to(device).long().view(-1)

            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            
            l = criterion(outputs, labels)
            l.backward()

            loss += l
            correct += (
                torch.max(outputs, 1)[1] == labels.long().view(-1)
            ).sum().item()

            self.optimizer.step()
        return loss.item(), correct * 100 / len(dataloader.dataset)

    def name(self):
        return f'EEGNet-{self.acti().__str__().split("(")[0]}-{self.dropout}'

    def load(self):
        self.load_state_dict(torch.load('weight/' + self.name()))
        return self


