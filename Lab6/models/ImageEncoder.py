import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):

    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.image_encoder = nn.Sequential(    
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512)
        )

    def forward(self, x):
        image_features = self.image_encoder(x)
        pooled_features = torch.sum(image_features, dim=(2, 3))
        return image_features, pooled_features
