import sys
sys.path.append('../../..')

from dataset.generator import TwoCircles

from model.optimizer import Adam

from model.function.loss import BinaryCrossEntropy

from model.function.activation import ReLu, SoftMax

from model.structure import FullyConnected


# dataset
dataset = TwoCircles()
dataset_num = 500

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
epochs = 30
optimizer = Adam(0.05, 0.99)
loss = BinaryCrossEntropy()
