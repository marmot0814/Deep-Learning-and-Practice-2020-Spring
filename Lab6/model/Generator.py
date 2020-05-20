import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config

from utils.func import weights_init

class Generator(nn.Module):

    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.d = d
        self.conv1_1 = nn.Conv2d(d, d*8, 3, 1, 1)
        self.conv1_1_bn = nn.BatchNorm2d(d*8)
        self.conv1_2 = nn.Conv2d(48, d*8, 3, 1, 1)
        self.conv1_2_bn = nn.BatchNorm2d(d*8)
        self.conv2 = nn.Conv2d(d*16, d*8, 3, 1, 1)
        self.conv2_bn = nn.BatchNorm2d(d*8)
        self.conv3 = nn.Conv2d(d*8, d*4, 3, 1, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*2, 3, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d*2)
        self.conv5 = nn.Conv2d(d*2, d, 3, 1, 1)
        self.conv5_bn = nn.BatchNorm2d(d)
        self.conv6 = nn.Conv2d(d, 3, 3, 1, 1)
        self.optim = torch.optim.Adam(self.parameters(), lr=config.lr_g, betas=(config.beta1, config.beta2))
        self.apply(weights_init)

    def forward(self, labels, c_embedding, s_embedding):
        (batch_sz, n_class) = labels.size()

        labels_embedded = []
        for label in labels:
            label_embedded = torch.tensor([]).to(config.device)
            for i in range(3):
                label_embedded = torch.cat([label_embedded, c_embedding(label[2 * i])])
                label_embedded = torch.cat([label_embedded, s_embedding(label[2 * i + 1])])
            labels_embedded.append(label_embedded)
        labels = torch.stack(labels_embedded).to(config.device)

        labels = labels.view(batch_sz, 48, 1, 1)
        noise = torch.rand((batch_sz, self.d, 1, 1)).to(config.device)

        x = F.relu(self.conv1_1_bn(self.conv1_1(nn.Upsample(scale_factor=2)(noise))))
        y = F.relu(self.conv1_2_bn(self.conv1_2(nn.Upsample(scale_factor=2)(labels))))
        x = torch.cat([x, y], 1)
        x = F.relu(self.conv2_bn(self.conv2(nn.Upsample(scale_factor=2)(x))))
        x = F.relu(self.conv3_bn(self.conv3(nn.Upsample(scale_factor=2)(x))))
        x = F.relu(self.conv4_bn(self.conv4(nn.Upsample(scale_factor=2)(x))))
        x = F.relu(self.conv5_bn(self.conv5(nn.Upsample(scale_factor=2)(x))))
        x = torch.tanh(self.conv6(nn.Upsample(scale_factor=2)(x)))
        return x

    def Train(self, images, labels, D, c_embedding, s_embedding):

        D.eval()
        self.train()
        g_loss = -D.Eval(self.forward(labels, c_embedding, s_embedding), labels, c_embedding, s_embedding).mean()

        self.optim.zero_grad()
        g_loss.backward()
        self.optim.step()
        return g_loss.item()

    def Eval(self, labels, c_embedding, s_embedding):
        return self.forward(labels, c_embedding, s_embedding)
