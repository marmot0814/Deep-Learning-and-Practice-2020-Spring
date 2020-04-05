import numpy as np

class FullyConnected:

    def __init__(self, m, n, minV = -5, maxV = 5):
        self.W = np.random.uniform(minV, maxV, (m + 1, n)).astype('float128')

    def forward(self, X):
        self.X = np.concatenate((
            X, np.array([ 1 for x in X ]).reshape(-1, 1)
        ), axis=1)
        return np.dot(self.X, self.W)

    def backward(self, E, optimizer):
        return optimizer.update(self, E)
