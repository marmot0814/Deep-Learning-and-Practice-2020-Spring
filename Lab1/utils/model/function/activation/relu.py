import numpy as np

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
