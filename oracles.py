import numpy as np
from scipy import sparse


class BaseSmoothOracle:
    """
    Base class for oracles.
    """
    def func(self, w):
        """
        Compute function value in point w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Compute gradient of function in point w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):

    def __init__(self, l2_coef=1.0):
        if l2_coef < 0:
            raise ValueError('l2 coefficient should be non-negative')

        self.l2_coef = l2_coef

    def func(self, X, y, w):
        y = y.copy()
        if np.any(np.unique(y) == 0):
            y[y == 1] = -1
            y[y == 0] = 1
        else:
            y *= -1

        return np.sum(np.logaddexp([0], (X @ w) * y)) / y.shape[0] + self.l2_coef * np.sum(w ** 2) / 2

    def grad(self, X, y, w):
        y = y.copy()
        if np.any(np.unique(y) == 0):
            y[y == 1] = -1
            y[y == 0] = 1
        else:
            y *= -1

        if isinstance(X, sparse.csr_matrix):
            ex = np.clip(np.exp((X @ w) * y), 1e-12, 1e+12)
            ex = y * ex / (ex + 1.0)
            return X.multiply(ex[:, np.newaxis]).sum(axis=0).A1 / y.shape[0] + self.l2_coef * w
        else:
            ex = np.clip(np.exp((X @ w) * y), 1e-12, 1e+12)
            ex = y * ex / (ex + 1.0)
            return np.sum(X * ex[:, np.newaxis], axis=0) / y.shape[0] + self.l2_coef * w
