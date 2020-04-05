import sys
sys.path.append('../../..')

from dataset.generator import XOR

from model.optimizer import SGD

from model.function.loss import MeanSquaredError

from model.function.activation import Sigmoid

from model.structure import FullyConnected


# dataset
dataset = XOR()
dataset_num = 50

# network
network_structure = [
    FullyConnected(2, 8),
    Sigmoid(),
    FullyConnected(8, 8),
    Sigmoid(),
    FullyConnected(8, 2),
    Sigmoid(),
]

# train
epochs = 200
optimizer = SGD(1)
loss = MeanSquaredError()
