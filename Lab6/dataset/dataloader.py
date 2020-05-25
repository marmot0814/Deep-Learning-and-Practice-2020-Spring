import torch
from torch.utils.data import Dataset
import json
import PIL
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np

class Dataset(Dataset):

    def __init__(self, path):
        self.labels = torch.load('dataset/labels.pth').long()
        self.images = torch.load('dataset/images.pth')

    def __getitem__(self, index):
        images, labels = [], []
        for i in range(3):
            images.append(self.images[index * 3 + i])
            labels.append(self.labels[index * 3 + i])
        images = torch.stack(images)
        labels = torch.stack(labels)
        return images, labels
        
    def __len__(self):
        return self.labels.size(0) // 3

class TestDataset(Dataset):

    def __init__(self, path):
        self.__load_objects(f'dataset/objects.json')
        self.__load_labels(path)

    def __load_objects(self, path):
        with open(path, 'r') as f:
            self.objects = json.load(f)

    def __load_labels(self, path):
        with open(path, 'r') as f:
            self.labels, self.onehot_labels = [], []
            for arr in json.load(f):
                label = torch.LongTensor([self.objects[item] for item in arr])
                onehot_label = torch.zeros(len(self.objects))
                onehot_label[label] = 1

                self.labels.append(label)
                self.onehot_labels.append(onehot_label)
            self.onehot_labels = torch.stack(self.onehot_labels)
