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
