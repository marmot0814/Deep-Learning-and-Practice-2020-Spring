import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 512
teacher_forcing_ratio = 0.5
LR = 0.01
MAX_LENGTH = 100
