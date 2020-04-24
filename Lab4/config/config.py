import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset_path = "dataset/train.json"
checkpoint_name = "net_params.pkl"

vocab_size = 30
embedding_size = 256
hidden_size = 256
num_layers = 1

teacher_forcing_ratio = 0.
