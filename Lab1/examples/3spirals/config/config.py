import sys
sys.path.append('../../..')

from dataset.generator import NSpirals

from model.optimizer import Adam

from model.function.loss import BinaryCrossEntropy

from model.function.activation import ReLu, SoftMax

from model.structure import FullyConnected


# dataset
dataset = NSpirals(3)
dataset_num = 600

# network
network_structure = [
    FullyConnected(2, 15, -1, 1),
    ReLu(0.1),
    FullyConnected(15, 15, -1, 1),
    ReLu(0.1),
    FullyConnected(15, 3, -1, 1),
    SoftMax()
]

# train
epochs = 200
optimizer = Adam(0.01, 0.99)
loss = BinaryCrossEntropy()
