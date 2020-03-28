import numpy as np
import matplotlib.pyplot as plt

from utils.data.displayer import line_plot, decision_region
from utils import *

class Network:

    def __init__(self):
        self.structure = []
        self.errors = []
        self.accuracies = []

        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(2, 4)

        self.acc_ax = fig.add_subplot(gs[0, 0:2])
        self.err_ax = fig.add_subplot(gs[1, 0:2])
        self.res_ax = fig.add_subplot(gs[0:2, 2:4])

        plt.get_current_fig_manager().full_screen_toggle()


    def add(self, layer):
        self.structure.append(layer)

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        for layer in self.structure:
            self.optimizer.initial(layer)
        self.loss = loss

    def train(self, X, Y, epochs = 10, ACC = None):
        plt.ion()
        acc, error = self.test(X, Y)
        self.optimizer.error = error
        print (f'epoch #0 - error: {error:.5f}, lr: {self.optimizer.lr:.5f}, acc: {acc * 100:.2f}%')

        for epoch in range(epochs):
            for x, y in zip(X, Y):
                x = x.reshape(1, -1)
                y = y.reshape(1, -1)
                y_hat = self.forward(x)
                self.backward(self.loss.gradient(y_hat, y))

            
            acc, error = self.test(X, Y)
            self.optimizer.tune(error)

            print (f'epoch #{epoch + 1} - error: {error:.5f}, lr: {self.optimizer.lr:.5f}, acc: {acc * 100:.2f}%')
            line_plot(self.acc_ax, self.accuracies, "accuracy", [0, 100])
            line_plot(self.err_ax, self.errors, "accuracy", [0, max(self.errors)])
            if epoch % 1 == 0:
                decision_region(self.res_ax, X, Y, self)

            if ACC != None and ACC <= acc:
                break
            plt.pause(0.1)
        plt.ioff()
        plt.show()


    def test(self, X, Y):

        Y_hat = self.forward(X)

        acc = np.sum(np.argmax(Y, axis=1) == np.argmax(Y_hat, axis=1)) / X.shape[0]
        error = np.mean(self.loss.evaluate(Y_hat, Y))

        self.accuracies.append(acc * 100)
        self.errors.append(error)

        return acc, error


    def forward(self, Y):
        for layer in self.structure:
            Y = layer.forward(Y)
        return Y

    def backward(self, E):
        for layer in self.structure[::-1]:
            E = layer.backward(E, self.optimizer)
