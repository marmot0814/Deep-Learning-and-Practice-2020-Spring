from dataset.dataloader import DataLoader

from model.Seq2Seq import Seq2Seq
from model.Encoder import EncoderRNN
from model.Decoder import DecoderRNN

from config.config import hidden_size, device

from utils.func import compute_bleu

result = []

dataloader = DataLoader('dataset/hidden_test/')

for hidden_size in [256, 512, 1024, 2048]:

    model = Seq2Seq(
        EncoderRNN(len(dataloader.dict.char2idx), hidden_size).to(device),
        DecoderRNN(hidden_size, len(dataloader.dict.char2idx)).to(device)
    ).load()

    res = model.test(dataloader, True)
    print (f'{model.name()} avg Bleu-4: {res:.5f}')
    result.append(res)

print ("model-256", result[0]);
print ("model-512", result[1]);
print ("model-1024", result[2]);
print ("model-2048", result[3]);

