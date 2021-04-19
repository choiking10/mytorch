import weakref
import numpy as np
import mytorch
from .variable import Variable
from .utils import as_array, as_variable


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]

        if mytorch.Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *x):
        raise NotImplementedError()

    def backward(self, *gy):
        raise NotImplementedError()

    def get_input_data(self):
        ret = [x.data for x in self.inputs]
        return ret if len(ret) > 1 else ret[0]

    def __gt__(self, other):
        return self.generation < other.generation


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.get_input_data()
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.get_input_data()
        gx = np.exp(x) * gy
        return gx


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.get_input_data()
        return x1 * gy, x0 * gy


class Neg(Function):
    def forward(self, x):
        y = -x
        return y

    def backward(self, gy):
        x = self.get_input_data()
        return -gy


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        x0, x1 = self.get_input_data()
        return gy, - gy


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.get_input_data()
        return gy / x1, - gy * x0 / (x1 ** 2)


class Pow(Function):
    def __init__(self, c):
        if isinstance(c, Variable):
            c = c.data
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x0 = self.get_input_data()
        c = self.c
        return gy * c * (x0 ** (c - 1))


def make_function(f, params=1, is_r=False):
    if params == 1:
        def operator_func(x):
            return f()(x)
        return operator_func
    if params == 2 and not is_r:
        def operator_func(x0, x1):
            x1 = as_variable(x1)
            return f()(x0, x1)
        return operator_func

    if params == 2 and is_r:
        def operator_func(x0, x1):
            x1 = as_variable(x1)
            return f()(x1, x0)
        return operator_func

    raise NotImplementedError()


square = make_function(Square)
exp = make_function(Exp)
add = make_function(Add, 2)
mul = make_function(Mul, 2)
neg = make_function(Neg)
sub = make_function(Sub, 2)
rsub = make_function(Sub, 2, True)
div = make_function(Div, 2)
rdiv = make_function(Div, 2, True)


def pow(x, c):
    return Pow(c)(x)


Variable.__add__ = add
Variable.__mul__ = mul
Variable.__radd__ = add
Variable.__rmul__ = mul
Variable.__neg__ = neg
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__truediv__ = div
Variable.__rturediv__ = rdiv
Variable.__pow__ = pow
