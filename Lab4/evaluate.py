from model.Encoder import Encoder
from model.Decoder import Decoder
from model.Seq2Seq import Seq2Seq

from dataset.DataHandler import DataHandler

from train import Trainer

from config import config


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
    trainer.load_model()

    print (trainer.evaluate(["crist", 'c'], 30))

if __name__ == '__main__':
    main()
