import torch
from data.dataloader import dataloader
from torchvision import transforms
from models import ResNet18, ResNet50
import pyprind
import torch.nn as nn

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = dataloader(
        "test",
        batch_size = 8,
        arg = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

    loss, acc = ResNet18(device, True).load().Test(test_loader, nn.CrossEntropyLoss(), device)

    print (f"Test Accuracy: {acc:.2f}%")

    loss, acc = ResNet50(device, True).load().Test(test_loader, nn.CrossEntropyLoss(), device)

    print (f"Test Accuracy: {acc:.2f}%")


if __name__ == '__main__':
    main()
