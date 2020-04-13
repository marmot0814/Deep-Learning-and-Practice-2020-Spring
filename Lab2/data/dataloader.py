import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def read_bci_data():
    S4b_train = np.load('data/S4b_train.npz')
    X11b_train = np.load('data/X11b_train.npz')
    S4b_test = np.load('data/S4b_test.npz')
    X11b_test = np.load('data/X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    return train_data, train_label, test_data, test_label

def gen_dataset(train_x, train_y, test_x, test_y):
    return [
        TensorDataset(
            torch.Tensor(x),
            torch.Tensor(y.reshape(-1, 1))
        ) for x, y in [(train_x, train_y), (test_x, test_y)]
    ]

def dataloader():
    train_dataset, test_dataset = gen_dataset(*read_bci_data())

    train_dataloader = DataLoader(train_dataset, len(train_dataset))
    test_dataloader = DataLoader(test_dataset, len(test_dataset))

    return train_dataloader, test_dataloader

if __name__ == '__main__':
    dataloader()
