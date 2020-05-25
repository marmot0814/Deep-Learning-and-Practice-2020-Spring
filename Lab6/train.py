import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import PIL
from PIL import Image

from dataset.dataloader import Dataset, TestDataset
from utils.evaluator import evaluation_model
from utils.func import make_grid
from models.GAN import GAN

from tqdm import tqdm

dataset = Dataset('dataset/')

test_dataset = TestDataset('dataset/test.json')

dataloader = DataLoader(
    dataset = dataset,
    batch_size = 64,
    shuffle=True,
    num_workers=8
)
torch.autograd.set_detect_anomaly(True)

model = GAN()

evaluator = evaluation_model()

for epoch in range(0, 100000):
    for batch, (images, labels) in enumerate(dataloader):
        model.train()
        images = images.permute((1, 0, 2, 3, 4)).cuda()
        labels = labels.permute((1, 0, 2)).cuda()
        model.Train(images, labels, epoch, batch)
        
        model.eval()
        images = model.Eval(test_dataset.labels)
        acc = int(evaluator.eval(images, test_dataset.onehot_labels) * 100)
        make_grid(images, epoch, batch, acc)
        if acc >= 70:
            model.save(epoch, batch)

