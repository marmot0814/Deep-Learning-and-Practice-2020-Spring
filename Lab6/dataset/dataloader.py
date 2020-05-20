import torch
from torch.utils.data import Dataset
import json
import PIL
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from config.config import device
import numpy as np

class Dataset(Dataset):

    def __init__(self, path):
        self.labels = torch.load(f'{path}labels.pth').long()
        self.images = torch.load(f'{path}images.pth')

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
        
    def __len__(self):
        return self.labels.size(0)

class TestDataset(Dataset):

    def __init__(self, path):
        self.__load_objects(f'{path}objects.json')
        self.__load_labels(f'{path}test.json')

    def __load_objects(self, path):
        with open(path, 'r') as f:
            self.objects = json.load(f)
        self.color, self.shape = {}, {}
        for key in self.objects.keys():
            c, s = key.split(' ')
            if not self.color.__contains__(c):
                idx = len(self.color) + 1
                self.color[c] = idx
            if not self.shape.__contains__(s):
                idx = len(self.shape) + 1
                self.shape[s] = idx

    def __load_labels(self, path):
        with open(path, 'r') as f:
            labels, onehot_labels = [], []
            for arr in json.load(f):
                label, ptr = torch.zeros(6), 0
                onehot_label = torch.zeros(len(self.objects))
                for item in arr:
                    c, s = item.split(' ')
                    label[ptr] = self.color[c]
                    ptr += 1
                    label[ptr] = self.shape[s]
                    ptr += 1
                    onehot_label[self.objects[item]] = 1

                labels.append(label)
                onehot_labels.append(onehot_label)
            self.labels = torch.stack(labels).long()
            self.onehot_labels = torch.stack(onehot_labels)
