import matplotlib.pyplot as plt
import numpy as np

TrainResNet18 = [73.34424712623225, 73.50795401971601, 73.50795401971601, 73.51151286522652, 73.50439517420548, 73.50439517420548, 73.49371863767394, 73.50083632869497, 73.51507171073703, 73.49727748318446]
TestResNet18 = [73.35231316725978, 73.33807829181495, 73.26690391459074, 73.22419928825623, 73.32384341637011, 73.23843416370107, 73.22419928825623, 73.20996441281139, 73.20996441281139, 73.25266903914591]
TrainResNet18_pre = [73.6787786042208, 75.68596747215203, 77.48674330047332, 78.49745542545998, 79.59713868820954, 80.21993665254992, 81.03491227445816, 81.87124096942952, 82.78230542012172, 83.54033951386171]
TestResNet18_pre = [74.69039145907473, 76.42704626334519, 77.90747330960853, 78.64768683274022, 79.1459074733096, 79.20284697508897, 79.41637010676156, 79.35943060498221, 79.68683274021352, 79.08896797153025]
TrainResNet50 = [73.45813018256878, 73.46168902807929, 73.44745364603723, 73.47592441012135, 73.44745364603723]
TrainResNet50_pre = [73.78554396953628, 76.04896971422471, 78.35510160503932, 79.83202249190363, 81.20929570447346]
TestResNet50 = [72.93950177935943, 73.32384341637011, 72.98220640569394, 73.29537366548043, 73.1814946619217]
TestResNet50_pre= [74.64768683274022, 77.73665480427046, 79.07473309608541, 79.55871886120997, 80.04270462633453]


def plot(train1, train2, test1, test2, title):
    fig, ax = plt.subplots()

    epoch = len(train1)

    ax.plot(
        range(1, epoch + 1),
        train1,
        label='Train(w/o pretraining)',
        color='green',
    )
    ax.plot(
        range(1, epoch + 1),
        test1,
        label='Test(w/o pretraining)',
        color='blue',
    )
    ax.plot(
        range(1, epoch + 1),
        train2,
        label='Train(with pretraining)',
        color='green',
        marker='o',
        markersize=4
    )
    ax.plot(
        range(1, epoch + 1),
        test2,
        label='Test(with pretraining)',
        color='blue',
        marker='o',
        markersize=4
    )

    ax.set_title(f"Result Comparison({title})")
    ax.legend(loc='upper left')
#ax.set_axisbelow(True)
    ax.yaxis.grid(color='#ddd')
    ax.xaxis.grid(color='#ddd')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy(%)")

    plt.savefig(f'{title}.png')

plot(TrainResNet18, TrainResNet18_pre, TestResNet18, TestResNet18_pre, "ResNet18")
plot(TrainResNet50, TrainResNet50_pre, TestResNet50, TestResNet50_pre, "ResNet50")
