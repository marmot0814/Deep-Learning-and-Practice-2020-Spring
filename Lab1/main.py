import numpy as np
import random

from utils.data.generator import NSpirals, XOR, Linear
from utils.data.processor import OneHot, Shuffle
from utils.data.displayer import *

from utils.model.network import Network

from utils.model.structure import FullyConnected

from utils.model.function.activation import Sigmoid, ReLu, SoftMax
from utils.model.function.loss import MeanSquaredError, BinaryCrossEntropy

from utils.model.optimizer import Adam



def construct_network(D, C):

    network = Network()

    network.add(FullyConnected(D, 15))
    network.add(ReLu(0.1))
    network.add(FullyConnected(15, 15))
    network.add(ReLu(0.1))
    network.add(FullyConnected(15, C))
    network.add(SoftMax())

    network.compile(Adam(0.01), BinaryCrossEntropy())

    return network

def construct_data(num = 500):

    # generate XOR data
    X, Y, D, C = XOR().generate(num)

    # one-hot encode label
    Y = OneHot().encode(Y)

    # random shuffle data
    X, Y = Shuffle().random(X, Y)

    return X, Y, D, C

def main():

    # construct training data. (./main.py)
    X, Y, D, C = construct_data(250)

    # construct neural network. (./main.py)
    network = construct_network(D, C)

    # train the neural network. (./utils/model/network.py)
    network.train(X, Y, 500, None)

    # predict result. (./utils/model/network.py)
    Y_hat = network.forward(X)

    # plot ground_truth and predict result as figure. (./utils/data/displayer/plot.py)
    gt_pd(
        X,
        OneHot().decode(Y),
        OneHot().decode(Y_hat),
        network
    )

if __name__ == '__main__':
    main()
