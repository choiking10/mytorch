import numpy as np

from mytorch import Function
from mytorch import as_variable, Variable


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.get_input_data()
        return gy * cos(x)


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x = self.get_input_data()
        return gy * -sin(x)


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = as_variable(self.get_output_data())
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
        self.x_shape = None

    def forward(self, x: np.ndarray):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy: np.ndarray):
        return reshape(gy, self.x_shape)


def reshape(x: np.ndarray or Variable, shape):
    if x.shape == shape:
        return as_variable(x)
    else:
        return Reshape(shape)(x)
