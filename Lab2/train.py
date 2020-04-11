import torch
import torch.nn as nn
import torch.optim as optim
from data.dataloader import dataloader
from model.EEGNet import EEGNet
from model.DeepConvNet import DeepConvNet
import json 
import matplotlib.pyplot as plt
import numpy as np
import datetime

class Trainer(object):

    def train(self, models, train_dataloader, test_dataloader, epoch_size, criterion, title):

        with open('weight/record.json', 'r') as f:
            record = json.load(f)

        accs = np.zeros((len(models) * 2, epoch_size))

        for epoch in range(epoch_size):
            print (f'{epoch}/{epoch_size}')

            for idx, model in enumerate(models):
                train_loss, train_acc = model.Train(train_dataloader, criterion, self.device)
                test_loss, test_acc = model.Test(test_dataloader, criterion, self.device)

                if not record.__contains__(model.name()) or test_acc > record[model.name()]:
                    record[model.name()] = test_acc
                    torch.save(model.state_dict(), 'weight/' + model.name())
                    with open('weight/record.json', 'w') as f:
                        f.write(json.dumps(record, indent=2, sort_keys=True))

                accs[idx * 2][epoch] = train_acc
                accs[idx * 2 + 1][epoch] = test_acc

                print (f'{model.name():30} - train: {train_acc:.2f}% / {train_loss:.2f}, test: {test_acc:.2f}% / {test_loss:.2f}')
            print ("")

        self.gen_plot(accs, models, title)

    def to(self, device):
        self.device = device
        return self

    def gen_plot(self, accs, models, title):
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy(%)")
        for idx, data in enumerate(accs):
            plt.plot(data, label=models[idx // 2].name() + '_' + ("train" if idx % 2 == 0 else "test"))
        plt.legend(loc='lower right')
        plt.title(title)
        plt.savefig("plot/" + datetime.datetime.now().__str__() + ".png")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    while True:
        models = [
            DeepConvNet(nn.LeakyReLU, 0.3).to(device),
            DeepConvNet(nn.LeakyReLU, 0.31).to(device),
            DeepConvNet(nn.LeakyReLU, 0.32).to(device),
            DeepConvNet(nn.LeakyReLU, 0.33).to(device),
            DeepConvNet(nn.LeakyReLU, 0.34).to(device),
            DeepConvNet(nn.LeakyReLU, 0.35).to(device),
            DeepConvNet(nn.LeakyReLU, 0.36).to(device),
            DeepConvNet(nn.LeakyReLU, 0.37).to(device),
            DeepConvNet(nn.LeakyReLU, 0.38).to(device),
            DeepConvNet(nn.LeakyReLU, 0.39).to(device),
            DeepConvNet(nn.LeakyReLU, 0.4).to(device),
            DeepConvNet(nn.LeakyReLU, 0.41).to(device),
            DeepConvNet(nn.LeakyReLU, 0.42).to(device),
            DeepConvNet(nn.LeakyReLU, 0.43).to(device),
            DeepConvNet(nn.LeakyReLU, 0.44).to(device),
            DeepConvNet(nn.LeakyReLU, 0.45).to(device),
            DeepConvNet(nn.LeakyReLU, 0.46).to(device),
            DeepConvNet(nn.LeakyReLU, 0.47).to(device),
            DeepConvNet(nn.LeakyReLU, 0.48).to(device),
            DeepConvNet(nn.LeakyReLU, 0.49).to(device),
            DeepConvNet(nn.LeakyReLU, 0.5).to(device),
        ]

        train_dataloader, test_dataloader = dataloader()

        Trainer().to(device).train(
            models = models,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            epoch_size=1000,
            criterion=nn.CrossEntropyLoss(),
            title="Activation function comparision(EEGNet)"
        )

if __name__ == '__main__':
    main()
