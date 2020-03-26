import numpy as np

class MeanSquaredError:

    def evaluate(self, Y_hat, Y):
        return (Y - Y_hat) ** 2 / 2.0

    def gradient(self, Y_hat, Y):
        return - (Y - Y_hat)

