import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config

from utils.func import weights_init

class Discriminator(nn.Module):

    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d( 3, d//2, 3, 2, 1)
        self.conv1_1_in = nn.InstanceNorm2d(d//2)
        self.conv1_2 = nn.Conv2d(48, d//2, 3, 2, 1)
        self.conv1_2_in = nn.InstanceNorm2d(d//2)
        self.conv2 = nn.Conv2d(d, d*2, 3, 2, 1)
        self.conv2_in = nn.InstanceNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 3, 2, 1)
        self.conv3_in = nn.InstanceNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 3, 2, 1)
        self.conv4_in = nn.InstanceNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 3, 2, 0)
        self.optim = torch.optim.Adam(self.parameters(), lr=config.lr_d, betas=(config.beta1, config.beta2))
        self.apply(weights_init)

    def forward(self, images, labels, c_embedding, s_embedding):
        (batch_sz, n_class), image_sz = labels.size(), images.size(2)

        labels_embedded = []
        for label in labels:
            label_embedded = torch.tensor([]).to(config.device)
            for i in range(3):
                label_embedded = torch.cat([label_embedded, c_embedding(label[2 * i])])
                label_embedded = torch.cat([label_embedded, s_embedding(label[2 * i + 1])])
            labels_embedded.append(label_embedded)
        labels = torch.stack(labels_embedded).to(config.device)

        labels = labels.view(batch_sz, 48, 1, 1).expand(batch_sz, 48, image_sz, image_sz)

        x = F.leaky_relu(self.conv1_1_in(self.conv1_1(images)), 0.2)
        y = F.leaky_relu(self.conv1_2_in(self.conv1_2(labels)), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_in(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_in(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_in(self.conv4(x)), 0.2)
        return self.conv5(x).view(batch_sz, -1)

    def Train(self, real_images, labels, G, c_embedding, s_embedding):
        (batch_size, n_class), image_sz = labels.size(), real_images.size(2)
        G.eval()
        self.train()

        fake_images = G.Eval(labels, c_embedding, s_embedding)
        f_s = self.Eval(fake_images, labels, c_embedding, s_embedding)
        r_s = self.Eval(real_images, labels, c_embedding, s_embedding)

        d_loss  = f_s.mean() - r_s.mean() + self.__gradient_penalty(real_images, fake_images, labels, c_embedding, s_embedding)

        self.optim.zero_grad()
        d_loss.backward()
        self.optim.step()
        return d_loss.item()
        
    def Eval(self, images, labels, c_embedding, s_embedding):
        return self.forward(images, labels, c_embedding, s_embedding)

    def __gradient_penalty(self, real_images, fake_images, labels, c_embedding, s_embedding):
        batch_size = labels.size(0)
        eps = torch.rand(batch_size, 1, 1, 1).expand_as(real_images).to(config.device)

        interpolation = eps * real_images + (1 - eps) * fake_images

        gradients = torch.autograd.grad(
            outputs = self.Eval(interpolation, labels, c_embedding, s_embedding).view(-1),
            inputs = interpolation,
            grad_outputs = torch.ones((batch_size)).to(config.device),
            create_graph=True,
            retain_graph=True
        )[0].view(batch_size, -1)

        return config.gamma * ((torch.sqrt(torch.sum(gradients ** 2, dim=1)) - 1) ** 2).mean()
