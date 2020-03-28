import numpy as np

class Adam:

    def __init__(self, lr = 0.01, decay = 0.99, beta_1 = 0.9, beta_2 = 0.999):
        self.lr = lr
        self.decay = decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.error = 1e9

    def initial(self, layer):
        if hasattr(layer, 'W'):
            layer.M = np.full(layer.W.shape, np.finfo(np.float).eps)
            layer.V = np.full(layer.W.shape, np.finfo(np.float).eps)

    def update(self, layer, E):
        gradient = np.dot(layer.X.T, E)

        layer.M = self.beta_1 * layer.M + (1 - self.beta_1) * gradient
        layer.V = self.beta_2 * layer.V + (1 - self.beta_2) * gradient ** 2

        M_hat = layer.M / (1 - self.beta_1)
        V_hat = layer.V / (1 - self.beta_2)

        E = np.dot(E, layer.W.T)[:, :-1]

        layer.W -= self.lr * M_hat / np.sqrt(V_hat)

        return E

    def tune(self, error):
        if self.error < error:
            self.lr *= self.decay * (1 - (error - self.error) / error) ** 2
        else:
            self.lr /= self.decay
        self.error = 0.2 * self.error + 0.8 * error
