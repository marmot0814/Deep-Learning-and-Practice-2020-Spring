import torch
import torch.nn as nn

import torchvision

def make_grid(images, epoch, batch, acc):
    grid = torchvision.utils.make_grid((images + 1) / 2, nrow=8)
    torchvision.utils.save_image(grid, f'plot/Epoch-{epoch}-{batch}-{acc}.png')

