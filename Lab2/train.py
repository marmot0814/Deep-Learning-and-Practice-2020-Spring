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
                train_loss, train_correct = model.Train(train_dataloader, criterion, self.device)
                test_loss, test_correct = model.Test(test_dataloader, criterion, self.device)

                if not record.__contains__(model.name()) or test_correct * 100 / len(test_dataloader.dataset) > record[model.name()]:
                    record[model.name()] = test_correct * 100 / len(test_dataloader.dataset)
                    torch.save(model.state_dict(), 'weight/' + model.name())
                    with open('weight/record.json', 'w') as f:
                        f.write(json.dumps(record, indent=2))

                accs[idx * 2][epoch] = train_correct*100/len(train_dataloader.dataset)
                accs[idx * 2 + 1][epoch] = test_correct*100/len(test_dataloader.dataset)

                print (f'{model.name():30} - train: {train_correct*100/len(train_dataloader.dataset):.2f}% / {train_loss:.2f}, test: {test_correct*100/len(test_dataloader.dataset):.2f}% / {test_loss:.2f}')

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

    models = [
        EEGNet(nn.ReLU, 0.7).to(device),
        EEGNet(nn.LeakyReLU, 0.7).to(device),
        DeepConvNet(nn.ELU, 0.2).to(device)
    ]

    train_dataloader, test_dataloader = dataloader()

    Trainer().to(device).train(
        models = models,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epoch_size=10,
        criterion=nn.CrossEntropyLoss(),
        title="plot title"
    )

if __name__ == '__main__':
    main()
