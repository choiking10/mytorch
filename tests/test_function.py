import unittest

import numpy as np

from mytorch.variable import Variable
from mytorch.utils import numerical_diff
from mytorch import function as F


class FunctionTestMixin:
    def get_function(self):
        raise NotImplementedError()

    def get_forward_input_output(self):
        raise NotImplementedError()

    def get_backward_input_output(self):
        raise NotImplementedError()

    def test_forward(self):
        x, expected_y = self.get_forward_input_output()
        f = self.get_function()
        x = Variable(x)
        y = f(x)
        self.assertEqual(y.data, expected_y)

    def test_backward(self):
        x, expected_grad = self.get_backward_input_output()
        f = self.get_function()
        x = Variable(x)
        y = f(x)
        y.backward()
        self.assertEqual(x.grad, expected_grad)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        f = self.get_function()
        y = f(x)
        y.backward()
        num_grad = numerical_diff(f, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


class SquareTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return F.square

    def get_forward_input_output(self):
        return np.array(2.0), np.array(4.0)

    def get_backward_input_output(self):
        return np.array(3.0), np.array(6.0)


class ExpTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return F.exp

    def get_forward_input_output(self):
        x = np.array(2.0)
        y = np.array(np.exp(x))
        return x, y

    def get_backward_input_output(self):
        x = np.array(3.0)
        grad_y = np.array(np.exp(x))
        return x, grad_y
