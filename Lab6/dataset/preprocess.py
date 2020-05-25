import PIL
from PIL import Image

import torch
import json
from torchvision import transforms
from tqdm import tqdm


with open('objects.json', 'r') as f:
    objects = json.load(f)

with open('labels.json', 'r') as f:
    labels = json.load(f)

trans = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

torch.save(trans(PIL.Image.open('png/background.png').convert('RGB')), 'background.pth')

lbls = []
imgs = []
for idx in tqdm(range(6004)):
    for sub_idx in range(3):
        key = f'CLEVR_train_{idx:06}_{sub_idx}.png'
        lbls.append(torch.LongTensor([objects[labels[key][-1]]]))
        imgs.append(trans(PIL.Image.open(f'png/{key}').convert("RGB")))

lbls = torch.stack(lbls)
imgs = torch.stack(imgs)

torch.save(lbls, 'labels.pth')
torch.save(imgs, 'images.pth')
