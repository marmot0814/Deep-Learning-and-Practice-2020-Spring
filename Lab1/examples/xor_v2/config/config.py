import sys
sys.path.append('../../..')

from dataset.generator import XOR

from model.optimizer import Adam

from model.function.loss import BinaryCrossEntropy

from model.function.activation import ReLu, SoftMax

from model.structure import FullyConnected


# dataset
dataset = XOR()
dataset_num = 50

# network
network_structure = [
    FullyConnected(2, 5),
    ReLu(0.1),
    FullyConnected(5, 5),
    ReLu(0.1),
    FullyConnected(5, 2),
    SoftMax()
]

# train
epochs = 100
optimizer = Adam(0.1, 0.99)
loss = BinaryCrossEntropy()
