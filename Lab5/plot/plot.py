import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import json
import numpy as np


def draw_plot(path, weight, title):
    with open(path, 'r') as f:
        log = json.load(f)

    x = np.arange(0., 100., 1/1227/4)
    y = weight(x)
    plt.plot(x, y, linewidth=0.5, alpha=0.3, c='red', label='KLD weight')
    plt.plot(range(0, 100), [l['BLEU'] for l in log], linewidth=0.5, label='BLEU-4 Score', c='blue')
    plt.plot(range(0, 100), [l['CrossEntropy_loss'] for l in log], linewidth=0.5, label='CrossEntropy Loss', c='orange')
    plt.plot(range(0, 100), [l['KL_loss'] for l in log], linewidth=0.5, label='KL Loss', c='green')
    plt.plot([0, 100], [0.7, 0.7], c='black', linestyle='dashed', linewidth=0.5)
    plt.xlabel('Epoch')
    plt.legend()
    plt.title(title)

    plt.show()

def monotonic(x):
    x = x * 1227 * 4 / 5000
    x[x > 1] = 1
    return x

def cyclical(x):
    x = x * 1227 * 4 % 10000 / 5000
    x[x > 1] = 1
    return x

draw_plot("monotonic.json", monotonic, "monotonic")
draw_plot("cyclical.json", cyclical, "cyclical")






'''
plt.plot(range(1, sz + 1), LOSS256, label='loss-256', color='r', alpha=0.5, linewidth=0.5)
plt.plot(range(1, sz + 1), LOSS512, label='loss-512', color='orange', alpha=0.5, linewidth=0.5)
plt.plot(range(1, sz + 1), LOSS1024, label='loss-1024', color='b', alpha=0.5, linewidth=0.5)
plt.plot(range(1, sz + 1), LOSS2048, label='loss-2048', color='g', alpha=0.5, linewidth=0.5)
plt.ylabel('CrossEntropyLoss')
plt.legend()
plt.title("loss plot")
plt.show()

plt.plot([1, sz], [80, 80], linestyle='--', linewidth=0.5, color='#000')
plt.plot([1, sz], [70, 70], linestyle='--', linewidth=0.5, color='#000')
plt.plot([1, sz], [60, 60], linestyle='--', linewidth=0.5, color='#000')
plt.plot([1, sz], [50, 50], linestyle='--', linewidth=0.5, color='#000')
plt.plot(range(1, sz + 1), BLEU256, label='bleu-256', color='r', alpha=0.5, linewidth=0.5)
plt.plot(range(1, sz + 1), BLEU512, label='bleu-512', color='orange', alpha=0.5, linewidth=0.5)
plt.plot(range(1, sz + 1), BLEU1024, label='bleu-1024', color='b', alpha=0.5, linewidth=0.5)
plt.plot(range(1, sz + 1), BLEU2048, label='bleu-2048', color='g', alpha=0.5, linewidth=0.5)
plt.xlabel('iteration (K)')
plt.ylabel('BLEU-4 score')
plt.legend()
plt.title("BLEU score plot")
plt.show()
'''
