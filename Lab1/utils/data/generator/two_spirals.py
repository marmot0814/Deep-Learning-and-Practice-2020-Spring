import numpy as np

class TwoSpirals:

    def generate(self, num):

        s = np.linspace(0, 20, num)

        x1, y1 = [  s / 4        * np.cos(s),  s / 4        * np.sin(s) ]
        x2, y2 = [ (s / 4 + 0.8) * np.cos(s), (s / 4 + 0.8) * np.sin(s) ]

        X = np.concatenate((np.array([x1, y1]), np.array([x2, y2])), axis=1).T
        Y = np.concatenate((np.zeros(num, dtype=int), np.ones(num, dtype=int))).reshape(-1, 1)

        return X, Y
