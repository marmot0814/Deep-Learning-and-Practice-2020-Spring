import sys
sys.path.append('../..')

import numpy as np
import random

from dataset.processor import OneHot, Shuffle
from dataset.displayer import gt_vs_pd

from model.network import Network

from config import config as config

def construct_network():

    network = Network(config.network_structure)

    network.compile(config.optimizer, config.loss)

    return network

def construct_dataset():

    # generate XOR dataset
    X, Y, D, C = config.dataset.generate(config.dataset_num)

    # one-hot encode label
    Y = OneHot().encode(Y)

    # random shuffle dataset
    X, Y = Shuffle().random(X, Y)

    return X, Y

def main():

    # construct training dataset. (./main.py)
    X, Y = construct_dataset()

    # construct neural network. (./main.py)
    network = construct_network()

    # train the neural network. (./model/network.py)
    network.train(X, Y, config.epochs)

    # predict result. (./model/network.py)
    Y_hat = network.forward(X)

    # plot ground_truth and predict result as figure. (./dataset/displayer/plot.py)
    gt_vs_pd(
        X,
        OneHot().decode(Y),
        OneHot().decode(Y_hat),
        network
    )

if __name__ == '__main__':
    main()
