import numpy as np

class Network:

    def __init__(self):
        self.structure = []

    def add(self, layer):
        self.structure.append(layer)

    def compile(self, optimizer):
        self.optimizer = optimizer
        for layer in self.structure:
            self.optimizer.initial(layer)

    def train(self, X, Y, loss, epochs = 10, ACC = None):
        for epoch in range(epochs):
            error = 0
            for x, y in zip(X, Y):
                x = x.reshape(1, -1)
                y = y.reshape(1, -1)
                y_hat = self.forward(x)
                error += np.mean(loss.evaluate(y_hat, y))
                self.backward(loss.gradient(y_hat, y))

            error /= X.shape[0]
            self.optimizer.tune(error)

            Y_hat = self.forward(X)
            acc = 100 * (X.shape[0] - np.sum(np.abs(np.argmax(Y, axis=1) - np.argmax(Y_hat, axis=1)))) / X.shape[0]

            print (f'epoch #{epoch + 1} - error: {error:.5f}, lr: {self.optimizer.lr:.5f}, acc: {acc:.2f}%')
            if ACC != None and ACC <= acc:
                break

    def forward(self, Y):
        for layer in self.structure:
            Y = layer.forward(Y)
        return Y

    def backward(self, E):
        for layer in self.structure[::-1]:
            E = layer.backward(E, self.optimizer)
