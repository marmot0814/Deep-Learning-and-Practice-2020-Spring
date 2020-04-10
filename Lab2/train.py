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

        accs = np.zeros((len(models) * 2, epoch_size + 1))

        for epoch in range(1, epoch_size + 1):

            train_correct, train_loss = [0 for m in models], [0 for m in models]

            for model in models:
                model.train()

            for idx, data in enumerate(train_dataloader):
                x, y = data
                inputs = x.to(self.device)
                labels = y.to(self.device).long().view(-1)

                for idx, model in enumerate(models):
                    model.optimizer.zero_grad()

                    outputs = model.forward(inputs)

                    loss = criterion(outputs, labels)
                    loss.backward()

                    train_correct[idx] += (
                        torch.max(outputs, 1)[1] == labels
                    ).sum().item()

                    train_loss[idx] += loss

                    model.optimizer.step()

            for model in models:
                model.eval()

            with torch.no_grad():

                print (f'{epoch}/{epoch_size}')

                test_correct, test_loss = [0 for m in models], [0 for m in models]

                for idx, data in enumerate(test_dataloader):
                    x, y = data
                    inputs = x.to(self.device)
                    labels = y.to(self.device).long().view(-1)

                    for idx, model in enumerate(models):
                        outputs = model.forward(inputs)

                        loss = criterion(outputs, labels)

                        test_correct[idx] += (
                            torch.max(outputs, 1)[1] == labels.long().view(-1)
                        ).sum().item()

                        test_loss[idx] += loss

                        print (f'{model.name():30} - train: {train_correct[idx]*100/len(train_dataloader.dataset):.2f}% / {train_loss[idx]:.2f}, test: {test_correct[idx]*100/len(test_dataloader.dataset):.2f}% / {test_loss[idx]:.2f}')

                        if not record.__contains__(model.name()) or test_correct[idx] * 100 / len(test_dataloader.dataset) > record[model.name()]:
                            record[model.name()] = test_correct[idx] * 100 / len(test_dataloader.dataset)
                            torch.save(model.state_dict(), 'weight/' + model.name())
                            with open('weight/record.json', 'w') as f:
                                f.write(json.dumps(record, indent=2))

                        accs[idx * 2][epoch] = train_correct[idx]*100/len(train_dataloader.dataset)
                        accs[idx * 2 + 1][epoch] = test_correct[idx]*100/len(test_dataloader.dataset)
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

def gen_dataset(train_x, train_y, test_x, test_y):
    return [
        TensorDataset(
            torch.stack(
                [torch.Tensor(x[i]) for i in range(x.shape[0])]
            ),
            torch.stack(
                [torch.Tensor(y[i:i+1]) for i in range(y.shape[0])]
            )
        ) for x, y in [(train_x, train_y), (test_x, test_y)]
    ]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [
        EEGNet(nn.ReLU, 0.7).to(device),
        EEGNet(nn.LeakyReLU, 0.7).to(device),
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
