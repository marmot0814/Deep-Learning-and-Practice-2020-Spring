from data.dataloader import dataloader
from model.EEGNet import EEGNet
from model.DeepConvNet import DeepConvNet
import torch
import torch.nn as nn


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader, test_dataloader = dataloader()

#    model = DeepConvNet(nn.LeakyReLU, 0.4).load(device)
    eegnet = EEGNet(nn.ReLU, 0.65).load(device)
    deepconvnet = DeepConvNet(nn.LeakyReLU, 0.4).load(device)
    loss, train_acc = eegnet.Test(train_dataloader, nn.CrossEntropyLoss(), device)
    loss, test_acc  = eegnet.Test(test_dataloader, nn.CrossEntropyLoss(), device)
    print ('-' * 63)
    print (f'{"Model":^30}||{"Train Acc(%)":^15}|{"Test Acc(%)":^15}')
    print ('-' * 63)
    print (f'{eegnet.name():^30}||{train_acc:^15.2f}|{test_acc:^15.2f}')
    loss, train_acc = deepconvnet.Test(train_dataloader, nn.CrossEntropyLoss(), device)
    loss, test_acc  = deepconvnet.Test(test_dataloader, nn.CrossEntropyLoss(), device)
    print (f'{deepconvnet.name():^30}||{train_acc:^15.2f}|{test_acc:^15.2f}')
    print ('-' * 63)

if __name__ == '__main__':
    main()
