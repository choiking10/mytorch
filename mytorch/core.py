from typing import List
import contextlib
import weakref
from queue import PriorityQueue

import numpy as np
import mytorch


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported.')
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0  # 세대 수를 기록하는 변수

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            self.grad = as_variable(np.ones_like(self.data))

        seen_set = set()
        funcs = PriorityQueue()
        funcs.put(self.creator)
        seen_set.add(self.creator)

        while funcs.qsize() != 0:
            f = funcs.get()
            gys = [output().grad for output in f.outputs]
            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None and x.creator not in seen_set:
                        funcs.put(x.creator)
                        seen_set.add(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def zerograd(self):
        self.grad = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            p = 'None'
        else:
            p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return f'variable({p})'

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return mytorch.functions.reshape(self, shape)

    def transpose(self):
        return mytorch.functions.transpose(self)

    @property
    def T(self):
        return self.transpose()


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [as_variable(y) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *x: np.ndarray):
        raise NotImplementedError()

    def backward(self, *gy: Variable):
        raise NotImplementedError()

    def get_input_data(self) -> Variable or List[Variable]:
        return self.inputs if len(self.inputs) > 1 else self.inputs[0]

    def get_output_data(self) -> Variable or List[Variable]:
        ret = [x() for x in self.outputs]
        return ret if len(ret) > 1 else ret[0]

    def __gt__(self, other):
        return self.generation < other.generation


class Parameter(Variable):
    pass


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(xs) -> Variable:
    if isinstance(xs, Variable):
        return xs
    return Variable(as_array(xs))


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.get_input_data()
        gx = 2 * x * gy
        return gx


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        x0, x1 = self.get_input_data()

        gx0, gx1 = gy, gy
        if x0.shape != x1.shape:
            gx0 = mytorch.functions.sum_to(gx0, x0.shape)
            gx1 = mytorch.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.get_input_data()
        gx0, gx1 = x1 * gy, x0 * gy
        if x0.shape != x1.shape:
            gx0 = mytorch.functions.sum_to(gx0, x0.shape)
            gx1 = mytorch.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


class Neg(Function):
    def forward(self, x):
        y = -x
        return y

    def backward(self, gy):
        return -gy


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        x0, x1 = self.get_input_data()
        gx0, gx1 = gy, -gy
        if x0.shape != x1.shape:
            gx0 = mytorch.functions.sum_to(gx0, x0.shape)
            gx1 = mytorch.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.get_input_data()
        gx0, gx1 = gy / x1, - gy * x0 / (x1 ** 2)
        if x0.shape != x1.shape:
            gx0 = mytorch.functions.sum_to(gx0, x0.shape)
            gx1 = mytorch.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


class Pow(Function):
    def __init__(self, c):
        if isinstance(c, Variable):
            c = c.data.item()
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x0 = self.get_input_data()
        c = self.c
        return gy * c * (x0 ** (c - 1))


def square(x):
    return Square()(x)


def add(x0, x1):
    x1 = as_variable(x1)
    return Add()(x0, x1)


def mul(x0, x1):
    x1 = as_variable(x1)
    return Mul()(x0, x1)


def neg(x):
    return Neg()(x)


def sub(x0, x1):
    x1 = as_variable(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_variable(x1)
    return Sub()(x1, x0)


def div(x0, x1):
    x1 = as_variable(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_variable(x1)
    return Div()(x1, x0)


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    Variable.__add__ = add
    Variable.__mul__ = mul
    Variable.__radd__ = add
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__getitem__ = mytorch.functions.get_item
