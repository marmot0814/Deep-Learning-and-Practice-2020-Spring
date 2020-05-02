import matplotlib.pyplot as plt

LOSS2048, LOSS1024, LOSS512, LOSS256 = [], [], [], []
BLEU2048, BLEU1024, BLEU512, BLEU256 = [], [], [], []

with open('2048result', 'r') as f:
    for line in f.readlines():
        loss, bleu = [*line.split(' ')]
        LOSS2048.append(float(loss))
        BLEU2048.append(float(bleu))
with open('1024result', 'r') as f:
    for line in f.readlines():
        loss, bleu = [*line.split(' ')]
        LOSS1024.append(float(loss))
        BLEU1024.append(float(bleu))

with open('512result', 'r') as f:
    for line in f.readlines():
        loss, bleu = [*line.split(' ')]
        LOSS512.append(float(loss))
        BLEU512.append(float(bleu))

with open('256result', 'r') as f:
    for line in f.readlines():
        loss, bleu = [*line.split(' ')]
        LOSS256.append(float(loss))
        BLEU256.append(float(bleu))

sz = len(LOSS512)

plt.plot(range(1, sz + 1), LOSS256, label='loss-256', color='r', alpha=0.5, linewidth=0.5)
plt.plot(range(1, sz + 1), LOSS512, label='loss-512', color='orange', alpha=0.5, linewidth=0.5)
plt.plot(range(1, sz + 1), LOSS1024, label='loss-1024', color='b', alpha=0.5, linewidth=0.5)
plt.plot(range(1, sz + 1), LOSS2048, label='loss-2048', color='g', alpha=0.5, linewidth=0.5)
plt.xlabel('iteration (K)')
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
