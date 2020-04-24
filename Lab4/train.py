from dataset.DataHandler import DataHandler

from model.Seq2Seq import Seq2Seq
from model.Encoder import Encoder
from model.Decoder import Decoder

from config import config

import torch
import torch.nn as nn

class Trainer:

    def __init__(self, model, data_handler, checkpoint_name):
        self.model = model
        self.data_handler = data_handler
        self.checkpoint_name = checkpoint_name

    def train(self, num_epochs, batch_size):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, num_epochs + 1):
            mini_batches = self.data_handler.mini_batches(batch_size)
            for i, o in mini_batches:
                ti = self.data_handler.encode(i)
                to = self.data_handler.encode(o)
                self.model(ti, to, config.teacher_forcing_ratio, optimizer, criterion)
                output = self.evaluate(i, 20)

                print (output)
                print (o)
            self.save_model()

    def save_model(self):
        torch.save(self.model.state_dict(), self.checkpoint_name)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.checkpoint_name))

    def evaluate(self, w, max_length):
        input = self.data_handler.encode(w)
        output = self.model.evaluate(input, max_length)
        output = self.data_handler.decode(output)
        return output

def main():
    data_handler = DataHandler(config.train_dataset_path)

    encoder = Encoder(
        data_handler.vocab_size,
        config.embedding_size,
        config.hidden_size,
        config.num_layers,
        config.device
    )
    decoder = Decoder(
        config.hidden_size,
        data_handler.vocab_size,
        config.num_layers,
        config.device
    )

    seq2seq = Seq2Seq(encoder, decoder, config.device)

    trainer = Trainer(seq2seq, data_handler, config.checkpoint_name)
    trainer.train(250, 1)

    trainer.evaluate(["crist"], 30)



if __name__ == "__main__":
    main()
