from dataset.dataloader import DataLoader

from model.Seq2Seq import Seq2Seq
from model.Encoder import EncoderRNN
from model.Decoder import DecoderRNN

from config.config import hidden_size, device

from utils.func import compute_bleu

hidden_size = 256

dataloader = DataLoader('dataset/official_test/')

model = Seq2Seq(
    EncoderRNN(len(dataloader.dict.char2idx), hidden_size).to(device),
    DecoderRNN(hidden_size, len(dataloader.dict.char2idx)).to(device)
).load()

print (f'Avg Bleu-4: {model.test(dataloader):.2f}%')

