import torch
import torch.nn as nn

from utils.func import compute_bleu

from config.config import device, MAX_LENGTH

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, target, use_teacher_forcing):
        output, hidden = self.encoder(input, self.encoder.initHidden())
        input = torch.tensor([[0]], device=device)
        ret = []
        for i in range(MAX_LENGTH):
            output, hidden = self.decoder(input, hidden)
            if use_teacher_forcing:
                if i == target.size(0):
                    break
                ret.append(output)
                input = target[i]
            else:
                ret.append(output)
                topv, topi = output.topk(1)
                input = topi.squeeze().detach()
                if input.item() == 1:
                    break
        return torch.stack(ret)

    def name(self):
        return self.encoder.name() + '-' + self.decoder.name()

    def load(self):
        self.load_state_dict(torch.load('weight/' + self.name()))
        return self

    def save(self):
        torch.save(self.state_dict(), 'weight/' + self.name()) 

    def predict(self, word, dataloader):
        word_tensor = dataloader.encode(word)
        output_tensor = self.forward(word_tensor, None, False).argmax(dim=2).view(-1, 1)
        output = dataloader.decode(output_tensor)
        return output

    def test(self, dataloader, display=False):
        total_bleu = 0
        for p in dataloader.test_data:
            output = self.predict(p[0], dataloader)
            total_bleu += compute_bleu(output, p[1])
            if not display or output == p[1]:
                continue
            print ('<', p[0])
            print ('=', p[1])
            print ('>', output)
        return total_bleu * 100 / len(dataloader.test_data)
