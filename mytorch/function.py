import numpy as np
from .variable import Variable


def numerical_diff(f, x, eps=1e-4):
    """
    수치 미분 구현부
    수치 미분은 구현하기 쉽고 거의 정확한 값을 얻을 수 있음. 이에 비해 역전파는 복잡한 알고리즘이라서
    구현하면 버그가 발생하기 쉬움. 따라서 역전파를 정확하게 구현하기 위해 수치 미분의 결과를 이용하곤 함.
    이를 gradient checking (기울기 확인) 이라고 함.
    :param f: 함수
    :param x: input 변수
    :param eps: 매우 작은 값
    :return: 미분의 결과물
    """
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class Function:
    def __call__(self, _input):
        x = _input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = _input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
