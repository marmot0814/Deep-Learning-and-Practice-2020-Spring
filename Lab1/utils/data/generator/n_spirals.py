import numpy as np
import matplotlib.pyplot as plt

class NSpirals:

    def generate(self, num, n):

        D = 2

        X = np.zeros((num * n, D))
        Y = np.zeros(num * n, dtype=int)
        
        for j in range(n):
            ix = range(num * j, num * (j + 1))
            r = np.linspace(10, 100, num)
            t = np.linspace(j * (2 * np.pi) / n, (j + 2 * n) * (2 * np.pi) / n, num)
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            Y[ix] = j

        return X, Y, D, n
