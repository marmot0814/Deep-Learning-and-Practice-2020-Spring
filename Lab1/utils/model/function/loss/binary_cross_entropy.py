import numpy as np

class BinaryCrossEntropy:

    def evaluate(self, Y_hat, Y):
        return - (Y * np.log(1e-9 + Y_hat) + (1 - Y) * np.log(1e-9 + 1 - Y_hat))

    def gradient(self, Y_hat, Y):
        return - (Y / (1e-9 + Y_hat) - (1 - Y) / (1e-9 + 1 - Y_hat))
