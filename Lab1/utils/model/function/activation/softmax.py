import numpy as np

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
