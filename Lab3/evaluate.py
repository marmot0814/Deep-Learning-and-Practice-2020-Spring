import torch
from data.dataloader import dataloader
from models import ResNet18, ResNet50
import pyprind

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = dataloader(16)
    
    model = ResNet18(device, True).load()

    model.eval()
    with torch.no_grad():
        correct = 0
        bar = pyprind.ProgPercent(len(test_loader), title="Testing...")
        for idx, data in enumerate(test_loader):
            x, y = data
            inputs = x.to(device).float()
            labels = y.to(device).long().view(-1)

            outputs = model(inputs)
            correct += (
                torch.max(outputs, 1)[1] == labels.long().view(-1)
            ).sum().item()
            bar.update(1)

    print (correct / len(test_loader.dataset))



if __name__ == '__main__':
    main()
