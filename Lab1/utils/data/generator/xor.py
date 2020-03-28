import numpy as np

class XOR:

    def generate(self, num):

        s = np.linspace(0, 1, num // 2)

        X = np.array([[x, x] for x in s] + [[x, 1 - x] for x in s])
        
        Y = np.array([1 * (x[0] == x[1]) for x in X]).reshape(-1, 1)

        return X, Y, 2, 2
