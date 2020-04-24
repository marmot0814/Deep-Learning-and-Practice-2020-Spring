import torch
import torch.nn as nn
import random

from torch.autograd import Variable

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, inputs, targets, teacher_forcing_ratio, optimizer, criterion):

        batch_size = inputs.size(1)
        hidden = self.encoder.initHidden(batch_size)

        optimizer.zero_grad()
        loss = 0

        encoder_outputs, encoder_hidden = self.encoder(inputs, hidden)

        decoder_input = self.decoder.initInputs(batch_size)
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        print ("==================================")
        print ("Training Part")
        print (use_teacher_forcing)
        for di in range(targets.size(0)):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            print ("input: ", decoder_input)
            print ("output: ", decoder_output)
            topv, topi = decoder_output.topk(1)
            print ("output_decoder: ", topi)
            print ("targets: ", targets[di])
            loss += criterion(decoder_output, targets[di])
            if use_teacher_forcing:
                decoder_input = targets[di].view(1, -1)
            else:
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.view(1, -1)

        loss.backward()
        optimizer.step()
        print (loss.item() / targets.size(0))

    def evaluate(self, inputs, max_length):
        print ("Evaluate part")
        batch_size = inputs.size(1)
        hidden = self.encoder.initHidden(batch_size)
        encoder_outputs, encoder_hidden = self.encoder(inputs, hidden)

        decoder_input = self.decoder.initInputs(batch_size)
        decoder_hidden = encoder_hidden

        decoder_outputs = Variable(torch.zeros(max_length, batch_size, device=self.device, dtype=torch.long))

        for t in range(max_length):
            decoder_output, deocder_hidden = self.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)

            print ("input: ", decoder_input)
            print ("output: ", decoder_output)
            print ("output_decoder: ", topi)

            decoder_outputs[t] = topi.view(1, -1)
            decoder_input = topi.view(1, -1).detach()
            if decoder_input.item() == 1:
                break
        return decoder_outputs
