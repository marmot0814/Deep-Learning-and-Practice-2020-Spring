import pandas as pd
from torch.utils.data import DataLoader
from torch.utils import data
from torchvision import transforms
import numpy as np
import cv2
import PIL


def getData(root, mode):
    if mode == 'train':
        img = pd.read_csv(root + 'train_img.csv')
        label = pd.read_csv(root + 'train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv(root + 'test_img.csv')
        label = pd.read_csv(root + 'test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)

def scaleRadius(img, scale):
    x = img[img.shape[0] // 2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    return cv2.resize(img, (0, 0), fx=s, fy=s)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode, arg=None):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(root, mode)
        self.mode = mode
        trans=[]
        if arg:
            trans += arg

        self.transforms = transforms.Compose(trans)
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """


        img = PIL.Image.open(self.root + 'jpeg/' + self.img_name[index] + '.jpeg')
        img = self.transforms(img)
        label = self.label[index]
        return img, label

def dataloader(batch_size):
    train_dataset = RetinopathyLoader('./data/', 'train', arg=[
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=8, 
        pin_memory=True,
        shuffle=True
    )
    test_dataset = RetinopathyLoader('./data/', 'test', arg=[
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
    return train_loader, test_loader

if __name__ == '__main__':
    d = RetinopathyLoader("./", "train")
    img, label = d[0]
    print (img.shape)
    print (label)
