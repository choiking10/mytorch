import numpy as np

from mytorch import Function, core
from mytorch import as_variable, Variable
from mytorch import utils


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


class Transpose(Function):
    def forward(self, x: np.ndarray):
        return x.transpose()

    def backward(self, gy: Variable):
        return transpose(gy)


def transpose(x):
    return Transpose()(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x: np.ndarray):
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy: Variable):
        x = self.get_input_data()
        gx = broadcast_to(gy, x.shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x: np.ndarray):
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy: Variable):
        x = self.get_input_data()
        gx = broadcast_to(gy, x.shape)
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class Sum(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x:np.ndarray):
        y = np.sum(x, axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy: Variable):
        x = self.get_input_data()
        gy = utils.reshape_sum_backward(gy, x.shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, x.shape)
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class MatMul(Function):
    def forward(self, x: np.ndarray, W: np.ndarray):
        y = x.dot(W)
        return y

    def backward(self, gy: Variable):
        x, W = self.get_input_data()
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    return MatMul()(x, W)


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy: Variable):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = - gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


class Linear(Function):
    def forward(self, x: np.ndarray, W: np.ndarray, b: np.ndarray):
        y = np.dot(x, W)
        if b is not None:
            y = y + b
        return y

    def backward(self, gy: Variable):
        x, W, b = self.get_input_data()
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        gb = sum_to(gy, b.shape) if b.data is not None else None
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b

    # t의 데이터는 + 연산시 역전파에 필요하지 않음.
    # matmul의 역전파 시 x, W, b 만 사용 따라서 t의 data는 필요하지 않음.
    t.data = None
    return y


class Sigmoid(Function):
    def forward(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def backward(self, gy: Variable):
        y = self.get_output_data()
        return gy * y * (1 - y)


def sigmoid(x):
    return Sigmoid()(x)
