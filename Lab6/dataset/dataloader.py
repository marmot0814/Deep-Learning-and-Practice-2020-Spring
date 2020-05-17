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
        self.labels = torch.load(f'{path}labels.pth')
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

    def __load_labels(self, path):
        with open(path, 'r') as f:
            labels = []
            for arr in json.load(f):
                label = torch.zeros(len(self.objects))
                for item in arr:
                    label[self.objects[item]] = 1
                labels.append(label)
            self.labels = torch.stack(labels)
