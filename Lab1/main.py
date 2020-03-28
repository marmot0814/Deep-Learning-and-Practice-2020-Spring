import numpy as np

from utils import *

import random

def construct_network(D, C):

    network = Network()

    network.add(FullyConnected(D, 5))
    network.add(ReLu(0.1))
    network.add(FullyConnected(5, 5))
    network.add(ReLu(0.1))
    network.add(FullyConnected(5, C))
    network.add(ReLu(0.1))
    network.add(SoftMax())

    network.compile(Adam(), BinaryCrossEntropy())

    return network

def construct_data(num = 500):

    X, Y, D, C = NSpirals().generate(num, 2)
#    X, Y, D, C = XOR().generate(num)

    # one hot
    Y = np.eye(num, np.max(Y) + 1)[Y].reshape(-1, np.max(Y) + 1)

    # random shuffle
    p = np.random.permutation(len(X))
    X = X[p]
    Y = Y[p]

    return X, Y, D, C

def main():
    X, Y, D, C = construct_data(400)

    network = construct_network(D, C)

    network.train(X, Y, 500, None)

    Y_hat = network.forward(X)
    gt_pd(
        X,
        np.argmax(Y,     axis=1).reshape(-1, 1),
        np.argmax(Y_hat, axis=1).reshape(-1, 1),
        network
    )


if __name__ == '__main__':
    main()
