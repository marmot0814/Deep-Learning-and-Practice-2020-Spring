import numpy as np

class FullyConnected:

    def __init__(self, m, n):
        self.W = np.random.uniform(-1, 1, (m + 1, n))

    def forward(self, X):
        self.X = np.concatenate((
            X, np.array([ 1 for x in X ]).reshape(-1, 1)
        ), axis=1)
        return np.dot(self.X, self.W)

    def backward(self, E, optimizer):
        return optimizer.update(self, E)
