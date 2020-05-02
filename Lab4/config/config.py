import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 1024
teacher_forcing_ratio = 0.5
MAX_LENGTH = 100
