import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import PIL
from PIL import Image

from dataset.dataloader import Dataset, TestDataset
from utils.evaluator import evaluation_model
from utils.func import make_grid
from config.config import device
from model.GAN import GAN


dataset = Dataset('dataset/')
test_dataset = TestDataset('dataset/')
dataloader = DataLoader(
    dataset = dataset,
    batch_size = 1024,
)

model = GAN(64).to(device)
evaluator = evaluation_model()

counter = 0
for epoch in range(100000):

    images = model.Eval(test_dataset.labels)
    make_grid(images, epoch, int(evaluator.eval(images, test_dataset.labels) * 100))
    model.save(epoch)

    for (images, labels) in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        model.Train(images, labels, counter)
        counter += 1

