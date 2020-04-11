from data.dataloader import dataloader
from model.EEGNet import EEGNet
from model.DeepConvNet import DeepConvNet
import torch
import torch.nn as nn


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader, test_dataloader = dataloader()

#    model = DeepConvNet(nn.LeakyReLU, 0.4).load(device)
    model = EEGNet(nn.ReLU, 0.7).load(device)
    loss, accuracy = model.Test(test_dataloader, nn.CrossEntropyLoss(), device)
    loss, accuracy = model.Test(train_dataloader, nn.CrossEntropyLoss(), device)
    print (loss, accuracy)
    

if __name__ == '__main__':
    main()
