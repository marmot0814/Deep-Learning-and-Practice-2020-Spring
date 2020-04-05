import numpy as np

class Sigmoid:

    def evaluate(self, X):
        return 1 / (1 + np.exp(-X))

    def gradient(self, X):
        return self.evaluate(X) * (1 - self.evaluate(X))

    def forward(self, X):
        self.X = X
        return self.evaluate(X)

    def backward(self, E, optimizer):
        return self.gradient(self.X) * E

class ReLu:

    def __init__(self, r):
        self.r = r

    def evaluate(self, X):
        return X * (X > 0) + self.r * (X < 0) * X

    def gradient(self, X):
        return 1 * (X > 0) + self.r * (X < 0)

    def forward(self, X):
        self.X = X
        return self.evaluate(X)

    def backward(self, E, optimizer):
        return self.gradient(self.X) * E

class SoftMax:

    def evaluate(self, X):
        X = X - np.max(X, axis=1).reshape(-1, 1)
        e_X = np.exp(X)
        return e_X / np.sum(e_X, axis=1).reshape(-1, 1)

    def gradient(self, X):
        return self.evaluate(X).T * (np.identity(X.shape[1]) - self.evaluate(X))

    def forward(self, X):
        self.X = X
        return self.evaluate(X)

    def backward(self, E, optimizer):
        return np.dot(E, self.gradient(self.X))
