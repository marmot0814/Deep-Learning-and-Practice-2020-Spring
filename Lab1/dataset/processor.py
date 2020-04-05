import numpy as np

class OneHot:

    def encode(self, X):

        assert len(X.shape) == 2 and X.shape[1] == 1

        return np.eye(np.max(X) + 1)[X.flatten()]

    def decode(self, X):

        assert len(X.shape) == 2

        return np.argmax(X, axis=1).reshape(-1, 1)

class Shuffle:

    def random(self, X, Y):

        assert X.shape[0] == Y.shape[0]

        p = np.random.permutation(len(X))

        return X[p], Y[p]
