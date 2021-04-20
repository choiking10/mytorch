import numpy as np
from mytorch import Variable


def numerical_gradient(f, *args, eps=1e-4):
    grads = []
    for X in args:
        grad = np.zeros_like(X.data)
        for idx, x in np.ndenumerate(X.data):
            X.data[idx] = x - eps
            y0 = f(*args)
            X.data[idx] = x + eps
            y1 = f(*args)
            X.data[idx] = x
            grad[idx] = (y1.data - y0.data) / (2 * eps)
        grads.append(grad)
    return grads


def as_tuple(x):
    if isinstance(x, list):
        return tuple(x)
    if not isinstance(x, tuple):
        return (x, )
    return x
