import torch
import torch.nn as nn
import torch.optim as optim
import data.dataloader as dataloader
from torch.utils.data import TensorDataset, DataLoader
from model.EEGNet import EEGNet
from model.DeepConvNet import DeepConvNet


class Trainer(object):

    def train(self, models, train_dataset, batch_size, test_dataset, epoch_size, criterion):

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))

        for epoch in range(epoch_size):

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

                print (f'{epoch + 1}/{epoch_size}')

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

                        print (f'{model.name():30} - train: {train_correct[idx]*100/len(train_dataset):.2f}% / {train_loss[idx]:.2f}, test: {test_correct[idx]*100/len(test_dataset):.2f}% / {test_loss[idx]:.2f}')
                    print ("")


    def to(self, device):
        self.device = device
        return self



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [
        DeepConvNet(nn.ELU, 0.75).to(device),
        DeepConvNet(nn.ReLU, 0.75).to(device),
        DeepConvNet(nn.LeakyReLU, 0.75).to(device),
    ]

    train_dataset, test_dataset = gen_dataset(*dataloader.read_bci_data())

    Trainer().to(device).train(
        models = models,
        train_dataset = train_dataset,
        batch_size=1080,
        test_dataset = test_dataset,
        epoch_size=2000,
        criterion=nn.CrossEntropyLoss()
    )


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


if __name__ == '__main__':
    main()
