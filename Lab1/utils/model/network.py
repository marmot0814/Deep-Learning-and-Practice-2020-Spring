import numpy as np
from utils import *

class Network:

    def __init__(self):
        self.structure = []
        self.errors = []
        self.accuracies = []

    def add(self, layer):
        self.structure.append(layer)

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        for layer in self.structure:
            self.optimizer.initial(layer)
        self.loss = loss

    def train(self, X, Y, epochs = 10, ACC = None):
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(2, 4)
        acc_ax = fig.add_subplot(gs[0, 0:2])
        err_ax = fig.add_subplot(gs[1, 0:2])
        res_ax = fig.add_subplot(gs[0:2, 2:4])
        plt.ion()
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        for epoch in range(epochs):
            error = 0
            for x, y in zip(X, Y):
                x = x.reshape(1, -1)
                y = y.reshape(1, -1)
                y_hat = self.forward(x)
                error += np.mean(self.loss.evaluate(y_hat, y))
                self.backward(self.loss.gradient(y_hat, y))

            acc, error = self.test(X, Y)
            self.optimizer.tune(error)

            print (f'epoch #{epoch + 1} - error: {error:.5f}, lr: {self.optimizer.lr:.5f}, acc: {acc * 100:.2f}%')
            self.accuracies.append(acc * 100)
            line_plot(acc_ax, self.accuracies, "accuracy", [0, epochs], [0, 100])
            self.errors.append(error)
            line_plot(err_ax, self.errors, "error", [0, epochs], [0, max(self.errors)])

            if epoch % 1 == 0:
                decision_region(res_ax, X, Y, self)

            if ACC != None and ACC <= acc:
                break
            plt.pause(0.1)
        plt.ioff()
        plt.show()


    def test(self, X, Y):
        Y_hat = self.forward(X)

        acc = np.sum(np.argmax(Y, axis=1) == np.argmax(Y_hat, axis=1)) / X.shape[0]
        error = np.mean(self.loss.evaluate(Y_hat, Y))

        return acc, error


    def forward(self, Y):
        for layer in self.structure:
            Y = layer.forward(Y)
        return Y

    def backward(self, E):
        for layer in self.structure[::-1]:
            E = layer.backward(E, self.optimizer)
