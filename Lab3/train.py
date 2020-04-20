from models import ResNet18, ResNet50
from data.dataloader import dataloader
from torchvision import transforms
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

        print (f'Epoch {epoch + 1}')
        for i, model in enumerate(models):
            loss, train_acc[i][epoch] = model.Train(train_loader, criterion, device)
            loss, test_acc[i][epoch]  = model.Test(test_loader, criterion, device)

            print (f"Train Acc: {train_acc[i][epoch]:.2f}%, Test Acc: {test_acc[i][epoch]:.2f}%")

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
    train_loader = dataloader(
        "train",
        batch_size = 8,
        arg = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    test_loader = dataloader(
        "test",
        batch_size = 8,
        arg = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

    while True:
        models = [
            ResNet50(device, pretrained=True).load(),
        ]


        train(
            models,
            train_loader,
            test_loader,
            epoch_size=10,
            criterion=nn.CrossEntropyLoss(),
            device=device,
        )

if __name__ == '__main__':
    main()
