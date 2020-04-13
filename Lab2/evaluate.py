from data.dataloader import dataloader
from model.EEGNet import EEGNet
from model.DeepConvNet import DeepConvNet
import torch
import torch.nn as nn


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader, test_dataloader = dataloader()

#    model = DeepConvNet(nn.LeakyReLU, 0.4).load(device)
    model1 = EEGNet(nn.ReLU, 0.65).load(device)
    model2 = DeepConvNet(nn.LeakyReLU, 0.4).load(device)
    loss, accuracy = model1.Test(test_dataloader, nn.CrossEntropyLoss(), device)
    print (f'{"model":<30} accuracy')
    print (f'{model1.name():<30} {accuracy:.2f}%')
    loss, accuracy = model2.Test(test_dataloader, nn.CrossEntropyLoss(), device)
    print (f'{model2.name():<30} {accuracy:.2f}%')
    

if __name__ == '__main__':
    main()
