import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 1e-4
lr_d = lr * 5
lr_g = lr

beta1 = 0.0
beta2 = 0.9
gamma = 10
n_critic = 1
