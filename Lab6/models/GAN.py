import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch import autograd

from models.gen.GenNet import GenNet
from models.dis.DisNet import DisNet
from models.ImageEncoder import ImageEncoder

import torchvision

class GAN(nn.Module):

    def __init__(self):
        super(GAN, self).__init__()

        self.layer_norm = nn.LayerNorm(256).cuda()

        self.gen = GenNet().cuda()
        self.gen_optim = optim.Adam(self.gen.parameters(), lr=0.0001, betas=(0, 0.9))

        self.dis = DisNet().cuda()
        self.dis_optim = optim.Adam(self.dis.parameters(), lr=0.0004, betas=(0, 0.9))

        self.rnn = nn.GRU(256, 256).cuda()
        self.rnn_optim = optim.Adam(self.rnn.parameters(), lr=0.0003)
        
        self.image_encoder = ImageEncoder().cuda()
        self.condi_encoder = nn.Embedding(24, 256).cuda()
        self.encoder_optim = optim.Adam(
            list(self.image_encoder.parameters()) + list(self.condi_encoder.parameters()),
        lr=6e-3)
        
        self.aux_criterion = nn.BCELoss()

    def Train(self, images, labels, epoch, batch):
        
        seq_len, batch_size, _, image_size, _ = images.size()

        prev_images = torch.load('dataset/background.pth').view(1, 3, image_size, image_size)
        prev_images = prev_images.expand(batch_size, 3, image_size, image_size).cuda()
        
        hidden = torch.zeros(1, batch_size, 256).cuda()
        
        res = []
        for t in range(seq_len):
            real_images = images[t]

            prev_fm, prev_vec = self.image_encoder(prev_images)
            real_fm, real_vec = self.image_encoder(real_images)

            condition = self.condi_encoder(labels[t]).view(1, batch_size, -1)
            onehot_labels = torch.zeros(batch_size, 24).cuda().scatter_(1, labels[t], 1)

            output, hidden = self.rnn(condition, hidden)
            output = self.layer_norm(output.view(batch_size, -1))

            fake_images = self.__forward_gen(output.detach(), prev_fm)

            d_loss, d_aux_loss = self.__optimize_dis(real_images, fake_images.detach(), prev_images, output, onehot_labels)
            g_loss, g_aux_loss = self.__optimize_gen(fake_images, prev_images.detach(), output.detach(), onehot_labels)
            print (f"d_loss: {d_loss:.4}, g_loss: {g_loss:.4}, real: {self.dis(real_images, output, prev_images)[0].mean().item():.4}, fake: {self.dis(fake_images, output, prev_images)[0].mean().item():.4}")
            
            prev_images = real_images
        self.__optimize_rnn()

    def __forward_gen(self, c, fm):
        noise = torch.FloatTensor(c.size(0), 100).normal_(0, 1).cuda()
        return self.gen(noise, c, fm)

    def __optimize_dis(self, real_images, fake_images, prev_images, condition, onehot_labels):

        real_images.requires_grad_()
        d_real, aux_real = self.dis(real_images, condition, prev_images)
        d_fake, aux_fake = self.dis(fake_images, condition, prev_images)

        d_loss = F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()
        aux_loss = self.aux_criterion(aux_real, onehot_labels) + self.aux_criterion(aux_fake, onehot_labels)
        d_loss += 5 * aux_loss
        d_loss += self.__gradient_penalty(d_real, real_images).mean()
        
        self.dis_optim.zero_grad()
        d_loss.backward(retain_graph=True)
        self.dis_optim.step()

        return d_loss.item(), aux_loss.item()

    def __gradient_penalty(self, d_real, real_images):
        batch_size = d_real.size(0)
        grad = autograd.grad(
            outputs = d_real.sum(),
            inputs = real_images,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        return (grad ** 2).view(batch_size, -1).sum(1)

    def __optimize_gen(self, fake_images, prev_images, condition, onehot_labels):
        d_fake, aux_fake = self.dis(fake_images, condition, prev_images)

        g_loss = -d_fake.mean()
        aux_loss = self.aux_criterion(aux_fake, onehot_labels)
        g_loss += 5 * aux_loss

        self.gen_optim.zero_grad()
        g_loss.backward(retain_graph=True)
        self.gen_optim.step()

        return g_loss.item(), aux_loss.item()

    def __optimize_rnn(self):
        self.rnn_optim.step()
        self.rnn.zero_grad()

        self.encoder_optim.step()
        self.image_encoder.zero_grad()
        self.condi_encoder.zero_grad()

    def Eval(self, labels):
        images = []
        for label in labels:
            prev_image = torch.load('dataset/background.pth').view(1, 3, 64, 64).cuda()
            hidden = torch.zeros(1, 1, 256).cuda()

            for item in label:
                item = item.view(1, 1).cuda()

                condition = self.condi_encoder(item).view(1, 1, -1)

                output, hidden = self.rnn(condition, hidden)
                output = self.layer_norm(output.view(1, -1))

                prev_fm, prev_vec = self.image_encoder(prev_image)

                prev_image = self.__forward_gen(output.detach(), prev_fm)

            images.append(prev_image)
        images = torch.cat(images, dim=0)
        return images

    def save(self, epoch, batch):
        torch.save(self.state_dict(), f'weight/Epoch-{epoch}-{batch}')

    def load(self, path):
        self.load_state_dict(torch.load(path))
        return self

