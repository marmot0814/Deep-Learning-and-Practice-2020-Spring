import numpy as np

class Linear:

    def generate(self, num):

        X = np.random.uniform(0, 1, (num, 2))

        Y = np.array([1 * (x[0] > x[1]) for x in X]).reshape(-1, 1)

        return X, Y, 2, 2

class XOR:

    def generate(self, num):

        s = np.linspace(0, 1, num // 2)

        X = np.array([[x, x] for x in s] + [[x, 1 - x] for x in s])
        
        Y = np.array([1 * (x[0] == x[1]) for x in X]).reshape(-1, 1)

        return X, Y, 2, 2

class NSpirals:

    def generate(self, num, n):

        D = 2

        X = np.zeros((num * n, D))
        Y = np.zeros((num * n, 1), dtype=int)
        
        for j in range(n):
            ix = range(num * j, num * (j + 1))
            r = np.linspace(10, 100, num)
            t = np.linspace(j * (2 * np.pi) / n, (j + 2 * n) * (2 * np.pi) / n, num)
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            Y[ix] = j

        return X, Y, D, n
