import sys
sys.path.append('../../..')

from dataset.generator import NSpirals

from model.optimizer import Adam

from model.function.loss import BinaryCrossEntropy

from model.function.activation import ReLu, SoftMax

from model.structure import FullyConnected


# dataset
dataset = NSpirals(7)
dataset_num = 1000

# network
network_structure = [
    FullyConnected(2, 15, -1, 1),
    ReLu(0.1),
    FullyConnected(15, 15, -1, 1),
    ReLu(0.1),
    FullyConnected(15, 7, -1, 1),
    SoftMax()
]

# train
epochs = 2000
optimizer = Adam(0.006, 0.99)
loss = BinaryCrossEntropy()
