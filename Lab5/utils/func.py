import time
import math
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import torch


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)

def KL_loss(m, lgv):
    return torch.mean(0.5 * (-lgv + (m ** 2) + torch.exp(lgv) - 1))

def KLD_weight(iteration):
    slope = 1 / 20000
    scope = (1 / slope) * 4
    return min(1, (iteration % scope) * slope)

def Teacher_forcing_ratio(epoch):
    return 0.6
    slope = 1 / 10
    return max(0, 1.0 - (slope * epoch))

def Gaussian_score(words):
    words_list = []
    score = 0
    yourpath = 'dataset/official/train.txt'
    with open(yourpath,'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score/len(words)
