import torch
from data.dataloader import dataloader
from torchvision import transforms
from models import ResNet18, ResNet50
import pyprind
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(df_confusion, title, cmap=plt.cm.Blues):
    fig, ax = plt.subplots()
    cax = ax.matshow(df_confusion, cmap=cmap)
    fig.colorbar(cax)
    
    for (i, j), z in np.ndenumerate(df_confusion):
        ax.text(j, i, f'{z:.2f}', ha='center', va='center')

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    plt.savefig("plot/ResNet50.png")

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

    y_actu = []
    for data in test_loader:
        x, y = data
        y_actu += [x.item() for x in y]

    resnet50 = ResNet50(device, True).load()
    y_pred = resnet50.Predict(test_loader, device)

    df_confusion = confusion_matrix(y_actu, y_pred)
    df_conf_norm = df_confusion / df_confusion.sum(axis=1).reshape(-1, 1)

    plot_confusion_matrix(df_conf_norm, 'Normalized Confusion Matrix(ResNet50)')


#    loss, acc = ResNet18(device, True).load().Test(test_loader, nn.CrossEntropyLoss(), device)

#    print (f"Test Accuracy: {acc:.2f}%")

#    loss, acc = ResNet50(device, True).load().Test(test_loader, nn.CrossEntropyLoss(), device)

#    print (f"Test Accuracy: {acc:.2f}%")

    


if __name__ == '__main__':
    main()
