import unittest

import numpy as np

from mytorch.variable import Variable
from mytorch.utils import numerical_diff, as_tuple, as_variable
from mytorch import function as F


class FunctionTestMixin:
    def get_function(self):
        raise NotImplementedError()

    def get_forward_input_output(self):
        raise NotImplementedError()

    def get_backward_input_output(self):
        raise NotImplementedError()

    def forward(self, *xs):
        f = self.get_function()
        y = f(*xs)
        return y

    def forward_and_backward(self, *xs):
        y = self.forward(*xs)
        y.backward()

    def test_forward(self):
        xs, expected_y = self.get_forward_input_output()
        xs = as_variable(xs)
        y = self.forward(*xs)
        self.assertEqual(y.data, expected_y)

    def test_backward(self):
        xs, expected_grads = self.get_backward_input_output()
        xs = as_variable(xs)
        expected_grads = as_tuple(expected_grads)
        self.forward_and_backward(*xs)
        for x, expected_grad in zip(xs, expected_grads):
            self.assertEqual(x.grad, expected_grad)

    # TODO: how could we test gradient check for multi input function?
    # def test_gradient_check(self):
    #     input_shape = self.get_forward_input_output()
    #     xs = Variable(np.random.rand(1))
    #
    #     xs = as_variable(xs)
    #     expected_grads = as_tuple(expected_grads)
    #     self.forward_and_backward(*xs)
    #
    #     num_grad = numerical_diff(f, x)
    #     flg = np.allclose(x.grad, num_grad)
    #     self.assertTrue(flg)


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


class AddTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return F.add

    def get_forward_input_output(self):
        x = np.array(2), np.array(3)
        y = np.array(5)
        return x, y

    def get_backward_input_output(self):
        x = np.array(2), np.array(3)
        grad_y = np.array(1), np.array(1)
        return x, grad_y
