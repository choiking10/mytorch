import mytorch
from mytorch import Function, core
from mytorch import as_variable, Variable
from mytorch import utils


class Sin(Function):
    def forward(self, x):
        xp = mytorch.cuda.get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, gy):
        x = self.get_input_data()
        return gy * cos(x)


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        xp = mytorch.cuda.get_array_module(x)
        y = xp.cos(x)
        return y

    def backward(self, gy):
        x = self.get_input_data()
        return gy * -sin(x)


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        xp = mytorch.cuda.get_array_module(x)
        y = xp.tanh(x)
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

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    else:
        return Reshape(shape)(x)


class Transpose(Function):
    def forward(self, x):
        return x.transpose()

    def backward(self, gy: Variable):
        return transpose(gy)


def transpose(x):
    return Transpose()(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        xp = mytorch.cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
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

    def forward(self, x):
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

    def forward(self, x):
        xp = mytorch.cuda.get_array_module(x)
        y = xp.sum(x, axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy: Variable):
        x = self.get_input_data()
        gy = utils.reshape_sum_backward(gy, x.shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, x.shape)
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class MatMul(Function):
    def forward(self, x, W):
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
    def forward(self, x, W, b):
        xp = mytorch.cuda.get_array_module(x)
        y = xp.dot(x, W)
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
    def forward(self, x):
        xp = mytorch.cuda.get_array_module(x)
        return 1 / (1 + xp.exp(-x))

    def backward(self, gy: Variable):
        y = self.get_output_data()
        return gy * y * (1 - y)


def sigmoid(x):
    return Sigmoid()(x)


class Exp(Function):
    def forward(self, x):
        xp = mytorch.cuda.get_array_module(x)
        return xp.exp(x)

    def backward(self, gy):
        x = self.get_input_data()
        gx = exp(x) * gy
        return gx


def exp(x):
    return Exp()(x)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x = self.get_input_data()
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


def get_item(x, slices):
    return GetItem(slices)(x)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = mytorch.cuda.get_array_module(x)
        gx = xp.zeros(self.in_shape)
        xp.add.at(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = mytorch.cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.get_output_data()
        gx = y * gy
        sumdx = sum(gx, axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        xp = mytorch.cuda.get_array_module(x)
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[xp.arange(N), t.ravel()]
        y = -log_p.sum() / xp.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape
        xp = mytorch.cuda.get_array_module(x)

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    acc = (pred == t.data).mean()
    return Variable(mytorch.core.as_array(acc))


class ReLU(Function):
    def forward(self, x):
        xp = mytorch.cuda.get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x = self.get_input_data()
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x):
    return ReLU()(x)
