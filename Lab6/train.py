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
    batch_size = 256,
)

model = GAN(128).to(device)
evaluator = evaluation_model()
start_from = 0



#start_from = 5
#model = model.load(start_from)

counter = 0
for epoch in range(start_from, 100000):
    for batch, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        model.Train(images, labels)
        counter += 1

        model.eval()
        images = model.Eval(test_dataset.labels)
        make_grid(images, epoch, batch, int(evaluator.eval(images, test_dataset.labels) * 100))
    model.save(epoch)
