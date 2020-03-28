import numpy as np

class Linear:

    def generate(self, num):

        X = np.random.uniform(0, 1, (num, 2))

        Y = np.array([
            1 * (x[0] > x[1]) for x in X
        ]).reshape(-1, 1)

        return X, Y, 2, 2
