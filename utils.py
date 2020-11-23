import numpy as np


def grad_finite_diff(function, w, eps=1e-8):
    grad = []
    for i in range(w.shape[0]):
        e_i = np.zeros(w.shape[0], dtype=np.float64)
        e_i[i] = 1.0
        grad.append((function(w + e_i * eps) - function(w)) / eps)

    return np.array(grad)
