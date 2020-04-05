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

    def __init__(self, n):
        self.n = n

    def generate(self, tot_num):

        D = 2

        num = tot_num // self.n

        X = np.zeros((num * self.n, D))
        Y = np.zeros((num * self.n, 1), dtype=int)
        
        for j in range(self.n):
            ix = range(num * j, num * (j + 1))
            r = np.linspace(10, 100, num)
            t = np.linspace(j * (2 * np.pi) / self.n, (j + 2 * self.n) * (2 * np.pi) / self.n, num)
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            Y[ix] = j

        return X, Y, D, self.n

class TwoCircles:

    def generate(self, n):

        D = 2

        num = n // 2

        t = np.linspace(0, 2 * np.pi, num)
        X = np.zeros((num * 2, D))
        Y = np.zeros((num * 2, 1), dtype=int)

        x0 = np.random.normal(0, 0.1, num)
        y0 = np.random.normal(0, 0.1, num)
        x1 = np.random.normal(0, 0.1, num)
        y1 = np.random.normal(0, 0.1, num)
        X[range(0, num)]   = np.c_[(1 + x0) * np.cos(t), (1 + y0) * np.sin(t)]
        Y[range(0, num)]   = 0

        X[range(num, 2 * num)] = np.c_[(0.5 + x1) * np.cos(t), (0.5 + y1) * np.sin(t)]
        Y[range(num, 2 * num)] = 1

        return X, Y, D, 2
        
