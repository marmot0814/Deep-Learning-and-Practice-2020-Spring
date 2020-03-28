import numpy as np

class MeanSquaredError:

    def evaluate(self, Y_hat, Y):
        return (Y - Y_hat) ** 2 / 2.0

    def gradient(self, Y_hat, Y):
        return - (Y - Y_hat)

class BinaryCrossEntropy:

    def evaluate(self, Y_hat, Y):
        return - (Y * np.log(1e-9 + Y_hat) + (1 - Y) * np.log(1e-9 + 1 - Y_hat))

    def gradient(self, Y_hat, Y):
        return - (Y / (1e-9 + Y_hat) - (1 - Y) / (1e-9 + 1 - Y_hat))
