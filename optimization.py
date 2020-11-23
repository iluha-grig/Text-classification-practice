import numpy as np
from scipy import sparse
import timeit
from . import oracles
import sys


class GDClassifier:

    def __init__(self, loss_function='binary_logistic', step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, **kwargs):
        self.loss_func = loss_function
        self.step_a = step_alpha
        self.step_b = step_beta
        self.tol = tolerance
        self.max_iter = max_iter
        self.coef_ = None
        if 'l2_coef' in kwargs:
            self.l2_coef = kwargs['l2_coef']
            self.bin_log = oracles.BinaryLogistic(self.l2_coef)
        else:
            self.bin_log = oracles.BinaryLogistic()

    def fit(self, X, y, w_0=None, trace=False, acc=None):
        if w_0 is None:
            w_0 = np.random.randn(X.shape[1])
        elif isinstance(w_0, list):
            w_0 = np.array(w_0)

        self.coef_ = w_0.copy()
        prev_loss_value = self.bin_log.func(X, y, self.coef_)
        if not trace:
            for i in range(self.max_iter):
                learning_rate = self.step_a / (i + 1) ** self.step_b
                self.coef_ -= learning_rate * self.bin_log.grad(X, y, self.coef_)
                new_loss_value = self.bin_log.func(X, y, self.coef_)
                if np.abs(prev_loss_value - new_loss_value) < self.tol:
                    break
                else:
                    prev_loss_value = new_loss_value
            else:
                print('Warning!!! Convergence not achieved!', file=sys.stderr)
            return None
        else:
            if acc is None:
                history = {'time': [0.0], 'func': [prev_loss_value]}
                for i in range(self.max_iter):
                    prev_time = timeit.default_timer()
                    learning_rate = self.step_a / (i + 1) ** self.step_b
                    self.coef_ -= learning_rate * self.bin_log.grad(X, y, self.coef_)
                    new_loss_value = self.bin_log.func(X, y, self.coef_)
                    history['func'].append(new_loss_value)
                    history['time'].append(timeit.default_timer() - prev_time)
                    if np.abs(prev_loss_value - new_loss_value) < self.tol:
                        break
                    else:
                        prev_loss_value = new_loss_value
                else:
                    print('Warning!!! Convergence not achieved!', file=sys.stderr)
                return history
            else:
                history = {'time': [0.0], 'func': [prev_loss_value], 'acc': []}
                history['acc'].append(np.sum(self.predict(acc[0]) == acc[1]) / acc[1].shape[0])
                for i in range(self.max_iter):
                    prev_time = timeit.default_timer()
                    learning_rate = self.step_a / (i + 1) ** self.step_b
                    self.coef_ -= learning_rate * self.bin_log.grad(X, y, self.coef_)
                    new_loss_value = self.bin_log.func(X, y, self.coef_)
                    history['func'].append(new_loss_value)
                    history['acc'].append(np.sum(self.predict(acc[0]) == acc[1]) / acc[1].shape[0])
                    history['time'].append(timeit.default_timer() - prev_time)
                    if np.abs(prev_loss_value - new_loss_value) < self.tol:
                        break
                    else:
                        prev_loss_value = new_loss_value
                else:
                    print('Warning!!! Convergence not achieved!', file=sys.stderr)
                return history

    def predict(self, X):
        if self.coef_ is None:
            raise Warning("Warning!!! Estimator wasn't fitted!")

        res = np.sign(X @ self.coef_)
        res[res == -1] = 0
        return res

    def predict_proba(self, X):
        if self.coef_ is None:
            raise Warning("Warning!!! Estimator wasn't fitted!")

        res = 1 / (1 + np.clip(np.exp((X @ self.coef_) * -1), 1e-12, 1e+12))
        return np.hstack(((1 - res)[:, np.newaxis], res[:, np.newaxis]))

    def get_objective(self, X, y):
        if self.coef_ is None:
            raise Warning("Warning!!! Estimator wasn't fitted!")

        return self.bin_log.func(X, y, self.coef_)

    def get_gradient(self, X, y):
        if self.coef_ is None:
            raise Warning("Warning!!! Estimator wasn't fitted!")

        return self.bin_log.grad(X, y, self.coef_)

    def get_weights(self):
        return self.coef_.copy()


class SGDClassifier(GDClassifier):

    def __init__(self, loss_function='binary_logistic', batch_size=10, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_epoch=1000, random_seed=1080, **kwargs):
        self.batch_size = batch_size
        self.random_seed = random_seed
        super(SGDClassifier, self).__init__(loss_function=loss_function, step_alpha=step_alpha, step_beta=step_beta,
                                            tolerance=tolerance, max_iter=max_epoch, **kwargs)

    def fit(self, X, y, w_0=None, trace=False, acc=None):
        if w_0 is None:
            w_0 = np.random.randn(X.shape[1])
        elif isinstance(w_0, list):
            w_0 = np.array(w_0)

        np.random.seed(self.random_seed)
        self.coef_ = w_0.copy()
        index = np.arange(y.shape[0])
        prev_loss_value = self.bin_log.func(X, y, self.coef_)
        d, r = y.shape[0] // self.batch_size, y.shape[0] % self.batch_size
        if not trace:
            for i in range(self.max_iter):  # epochs
                np.random.shuffle(index)
                X = X[index]
                y = y[index]
                for j in range(d):
                    learning_rate = self.step_a / (i * (d + int(r > 0)) + j + 1) ** self.step_b
                    self.coef_ -= learning_rate * self.bin_log.grad(X[j*self.batch_size:(j + 1)*self.batch_size],
                                                                    y[j*self.batch_size:(j + 1)*self.batch_size],
                                                                    self.coef_)
                if r > 0:
                    learning_rate = self.step_a / (i * (d + int(r > 0)) + d + 1) ** self.step_b
                    self.coef_ -= learning_rate * self.bin_log.grad(X[d * self.batch_size:],
                                                                    y[d * self.batch_size:],
                                                                    self.coef_)
                new_loss_value = self.bin_log.func(X, y, self.coef_)
                if np.abs(prev_loss_value - new_loss_value) < self.tol:
                    break
                else:
                    prev_loss_value = new_loss_value
            else:
                print('Warning!!! Convergence not achieved!', file=sys.stderr)
            return None
        else:
            if acc is None:
                history = {'time': [0.0], 'func': [prev_loss_value]}
                for i in range(self.max_iter):  # epochs
                    time1 = timeit.default_timer()
                    np.random.shuffle(index)
                    X = X[index]
                    y = y[index]
                    for j in range(d):
                        learning_rate = self.step_a / (i * (d + int(r > 0)) + j + 1) ** self.step_b
                        self.coef_ -= learning_rate * self.bin_log.grad(
                            X[j * self.batch_size:(j + 1) * self.batch_size],
                            y[j * self.batch_size:(j + 1) * self.batch_size],
                            self.coef_)
                    if r > 0:
                        learning_rate = self.step_a / (i * (d + int(r > 0)) + d + 1) ** self.step_b
                        self.coef_ -= learning_rate * self.bin_log.grad(X[d * self.batch_size:],
                                                                        y[d * self.batch_size:],
                                                                        self.coef_)
                    new_loss_value = self.bin_log.func(X, y, self.coef_)
                    history['func'].append(new_loss_value)
                    history['time'].append(timeit.default_timer() - time1)
                    if np.abs(prev_loss_value - new_loss_value) < self.tol:
                        break
                    else:
                        prev_loss_value = new_loss_value
                else:
                    print('Warning!!! Convergence not achieved!', file=sys.stderr)
                return history
            else:
                history = {'time': [0.0], 'func': [prev_loss_value], 'acc': []}
                history['acc'].append(np.sum(self.predict(acc[0]) == acc[1]) / acc[1].shape[0])
                for i in range(self.max_iter):  # epochs
                    time1 = timeit.default_timer()
                    np.random.shuffle(index)
                    X = X[index]
                    y = y[index]
                    for j in range(d):
                        learning_rate = self.step_a / (i * (d + int(r > 0)) + j + 1) ** self.step_b
                        self.coef_ -= learning_rate * self.bin_log.grad(
                            X[j * self.batch_size:(j + 1) * self.batch_size],
                            y[j * self.batch_size:(j + 1) * self.batch_size],
                            self.coef_)
                    if r > 0:
                        learning_rate = self.step_a / (i * (d + int(r > 0)) + d + 1) ** self.step_b
                        self.coef_ -= learning_rate * self.bin_log.grad(X[d * self.batch_size:],
                                                                        y[d * self.batch_size:],
                                                                        self.coef_)
                    new_loss_value = self.bin_log.func(X, y, self.coef_)
                    history['func'].append(new_loss_value)
                    history['acc'].append(np.sum(self.predict(acc[0]) == acc[1]) / acc[1].shape[0])
                    history['time'].append(timeit.default_timer() - time1)
                    if np.abs(prev_loss_value - new_loss_value) < self.tol:
                        break
                    else:
                        prev_loss_value = new_loss_value
                else:
                    print('Warning!!! Convergence not achieved!', file=sys.stderr)
                return history
