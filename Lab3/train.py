from models import ResNet18, ResNet50
from data.dataloader import dataloader
import torch
import time
import torch.nn as nn
import pyprind
import json
import PIL

def train(models, train_loader, test_loader, epoch_size, criterion, device):

    with open('weight/record.json') as f:
        record = json.load(f)

    train_acc = [[0 for x in range(epoch_size)] for x in range(len(models))]
    test_acc  = [[0 for x in range(epoch_size)] for x in range(len(models))]
    for epoch in range(epoch_size):

        bar = pyprind.ProgPercent(len(train_loader), title="Training epoch {} : ".format(epoch+1))
        for model in models:
            model.train()

        loss = 0
        for idx, data in enumerate(train_loader):
            x, y = data
            inputs = x.to(device).float()
            labels = y.to(device).long().view(-1)

            for i, model in enumerate(models):
                model.optimizer.zero_grad()
                outputs = model(inputs)

                t = criterion(outputs, labels)
                loss += t.item()
                t.backward()

                model.optimizer.step()

                train_acc[i][epoch] += (torch.max(outputs, 1)[1] == labels).sum().item()

            bar.update(1)

        print ("Train loss:", loss)

        for i in range(len(models)):
            train_acc[i][epoch] /= len(train_loader.dataset)


        for model in models:
            model.eval()

        with torch.no_grad():
            bar = pyprind.ProgPercent(len(test_loader), title="Testing epoch {} : ".format(epoch+1))
            loss = 0
            for idx, data in enumerate(test_loader):
                x, y = data
                inputs = x.to(device).float()
                labels = y.to(device).long().view(-1)

                for i, model in enumerate(models):

                    outputs = model(inputs)
                    loss += criterion(outputs, labels)
                    test_acc[i][epoch] += (torch.max(outputs, 1)[1] == labels).sum().item()

                bar.update(1)
            print ("Test loss:", loss.item())

        for i in range(len(models)):
            test_acc[i][epoch] /= len(test_loader.dataset)

        for i, model in enumerate(models):
            if not record.__contains__(model.name()) or test_acc[i][epoch] > record[model.name()]:
                record[model.name()] = test_acc[i][epoch]
                torch.save(model.state_dict(), 'weight/' + model.name())
                with open('weight/record.json', 'w') as f:
                    f.write(json.dumps(record, indent=2, sort_keys=True))

        with open('weight/log', 'a+') as f:
            for model in models:
                f.write(model.name())
                f.write("\n")
            f.write(train_acc.__str__())
            f.write("\n")
            f.write(test_acc.__str__())
            f.write("\n")
            f.write("\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = dataloader(8)

    while True:
        models = [
            ResNet18(device, pretrained=True).load(),
        ]

        train(
            models,
            train_loader,
            test_loader,
            epoch_size=10,
            criterion=nn.CrossEntropyLoss(),
            device=device
        )

if __name__ == '__main__':
    main()
