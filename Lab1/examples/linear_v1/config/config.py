import sys
sys.path.append('../../..')

from dataset.generator import Linear

from model.optimizer import SGD

from model.function.loss import MeanSquaredError

from model.function.activation import Sigmoid

from model.structure import FullyConnected


# dataset
dataset = Linear()
dataset_num = 100

# network
network_structure = [
    FullyConnected(2, 3),
    Sigmoid(),
    FullyConnected(3, 3),
    Sigmoid(),
    FullyConnected(3, 2),
    Sigmoid(),
]

# train
epochs = 50
optimizer = SGD(0.5)
loss = MeanSquaredError()
