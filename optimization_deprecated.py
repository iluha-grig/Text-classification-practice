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
                self.coef_ -= self.bin_log.grad(X, y, self.coef_) * learning_rate
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
                    self.coef_ = self.coef_ - learning_rate * self.bin_log.grad(X, y, self.coef_)
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
                    self.coef_ = self.coef_ - learning_rate * self.bin_log.grad(X, y, self.coef_)
                    new_loss_value = self.bin_log.func(X, y, self.coef_)
                    history['func'].append(new_loss_value)
                    history['time'].append(timeit.default_timer() - prev_time)
                    history['acc'].append(np.sum(self.predict(acc[0]) == acc[1]) / acc[1].shape[0])
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
                 tolerance=1e-5, max_iter=1000, random_seed=1080, **kwargs):
        self.batch_size = batch_size
        self.random_seed = random_seed
        super(SGDClassifier, self).__init__(loss_function=loss_function, step_alpha=step_alpha, step_beta=step_beta,
                                            tolerance=tolerance, max_iter=max_iter, **kwargs)

    def fit(self, X, y, w_0=None, trace=False, log_freq=1, acc=None):
        if w_0 is None:
            w_0 = np.random.randn(X.shape[1])
        elif isinstance(w_0, list):
            w_0 = np.array(w_0)

        np.random.seed(self.random_seed)
        self.coef_ = w_0.copy()
        index = np.random.choice(np.arange(y.shape[0]), self.batch_size, replace=False)
        prev_loss_value = self.bin_log.func(X[index], y[index], self.coef_)
        if not trace:
            for i in range(self.max_iter):
                learning_rate = self.step_a / (i + 1) ** self.step_b
                index = np.random.choice(np.arange(y.shape[0]), self.batch_size, replace=False)
                self.coef_ -= self.bin_log.grad(X[index], y[index], self.coef_) * learning_rate
                new_loss_value = self.bin_log.func(X[index], y[index], self.coef_)
                if np.abs(prev_loss_value - new_loss_value) < self.tol:
                    break
                else:
                    prev_loss_value = new_loss_value
            else:
                print('Warning!!! Convergence not achieved!', file=sys.stderr)
            return None
        else:
            if acc is None:
                history = {'time': [0.0], 'func': [prev_loss_value], 'epoch_num': [], 'weights_diff': []}
                epoch_num = 0.0
                prev_time = timeit.default_timer()
                old_coef = self.coef_
                for i in range(self.max_iter):
                    if i * self.batch_size / y.shape[0] - epoch_num > log_freq:
                        epoch_num = i * self.batch_size / y.shape[0]
                        history['epoch_num'].append(epoch_num)
                        history['func'].append(prev_loss_value)
                        history['weights_diff'].append(np.sum((self.coef_ - old_coef) ** 2))
                        old_coef = self.coef_
                        new_time = timeit.default_timer()
                        history['time'].append(new_time - prev_time)
                        prev_time = new_time
                    learning_rate = self.step_a / (i + 1) ** self.step_b
                    index = np.random.choice(np.arange(y.shape[0]), self.batch_size, replace=False)
                    self.coef_ -= self.bin_log.grad(X[index], y[index], self.coef_) * learning_rate
                    new_loss_value = self.bin_log.func(X[index], y[index], self.coef_)
                    if np.abs(prev_loss_value - new_loss_value) < self.tol:
                        break
                    else:
                        prev_loss_value = new_loss_value
                else:
                    print('Warning!!! Convergence not achieved!', file=sys.stderr)
                return history
            else:
                history = {'time': [0.0], 'func': [prev_loss_value], 'epoch_num': [], 'weights_diff': [], 'acc': []}
                history['acc'].append(np.sum(self.predict(acc[0]) == acc[1]) / acc[1].shape[0])
                epoch_num = 0.0
                prev_time = timeit.default_timer()
                old_coef = self.coef_
                for i in range(self.max_iter):
                    if i * self.batch_size / y.shape[0] - epoch_num > log_freq:
                        epoch_num = i * self.batch_size / y.shape[0]
                        history['epoch_num'].append(epoch_num)
                        history['func'].append(prev_loss_value)
                        history['acc'].append(np.sum(self.predict(acc[0]) == acc[1]) / acc[1].shape[0])
                        history['weights_diff'].append(np.sum((self.coef_ - old_coef) ** 2))
                        old_coef = self.coef_
                        new_time = timeit.default_timer()
                        history['time'].append(new_time - prev_time)
                        prev_time = new_time
                    learning_rate = self.step_a / (i + 1) ** self.step_b
                    index = np.random.choice(np.arange(y.shape[0]), self.batch_size, replace=False)
                    self.coef_ -= self.bin_log.grad(X[index], y[index], self.coef_) * learning_rate
                    new_loss_value = self.bin_log.func(X[index], y[index], self.coef_)
                    if np.abs(prev_loss_value - new_loss_value) < self.tol:
                        break
                    else:
                        prev_loss_value = new_loss_value
                else:
                    print('Warning!!! Convergence not achieved!', file=sys.stderr)
                return history
