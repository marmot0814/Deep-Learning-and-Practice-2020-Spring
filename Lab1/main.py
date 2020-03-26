import numpy as np

from utils import *

import random

def construct_network():

    network = Network()

    network.add(FullyConnected(2, 15))
    network.add(ReLu(0.1))
    network.add(FullyConnected(15, 15))
    network.add(ReLu(0.1))
    network.add(FullyConnected(15, 2))
    network.add(Sigmoid())

    network.compile(Adam())

    return network

def construct_data(num = 500):

    X, Y = TwoSpirals().generate(num)

    Y = np.eye(num, np.max(Y) + 1)[Y].reshape(-1, np.max(Y) + 1)

    p = np.random.permutation(len(X))
    X = X[p]
    Y = Y[p]

    return X, Y

def main():
    X, Y = construct_data(250)

    network = construct_network()

    network.train(X, Y, BinaryCrossEntropy(), 10000, 100)

    Y_hat = network.forward(X)
    gt_pd(
        X,
        np.argmax(Y,     axis=1).reshape(-1, 1),
        np.argmax(Y_hat, axis=1).reshape(-1, 1)
    )


if __name__ == '__main__':
    main()
