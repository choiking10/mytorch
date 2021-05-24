import unittest

import numpy as np

import mytorch.functions as F
from mytorch import as_variable
from tests.utils import FunctionTestMixin, ForwardAndBackwardCheckMixin


class SinTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return F.sin

    def get_forward_input_output(self):
        x = np.array(2.0)
        y = np.sin(x)
        return x, y

    def get_backward_input_output(self):
        x = np.array(3.0)
        grad = np.cos(x)
        return x, grad

    def test_numerical_check(self):
        self.numerical_gradient_check(1)


class CosTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return F.cos

    def get_forward_input_output(self):
        x = np.array(2.0)
        y = np.cos(x)
        return x, y

    def get_backward_input_output(self):
        x = np.array(3.0)
        grad = -np.sin(x)
        return x, grad

    def test_numerical_check(self):
        self.numerical_gradient_check(1)


class TanhTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return F.tanh

    def get_forward_input_output(self):
        x = np.array(2.0)
        y = np.tanh(x)
        return x, y

    def get_backward_input_output(self):
        x = np.array(3.0)
        y = np.tanh(x)
        grad = 1 - y * y
        return x, grad

    def test_numerical_check(self):
        self.numerical_gradient_check(1)


class ReshapeTest(unittest.TestCase, FunctionTestMixin):
    do_as_variable = False

    def get_function(self):
        return F.reshape

    def get_forward_input_output(self):
        x = as_variable(np.array([[1, 2, 3], [4, 5, 6]])), (6,)
        y = np.array([1, 2, 3, 4, 5, 6])
        return x, y

    def get_backward_input_output(self):
        x = as_variable(np.array([[1, 2, 3], [4, 5, 6]])), (6,)
        grad = np.array([[1, 1, 1], [1, 1, 1]])
        return x, grad


class TransposeTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return F.transpose

    def get_forward_input_output(self):
        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([[1, 4], [2, 5], [3, 6]])
        return x, y

    def get_backward_input_output(self):
        x = np.array([[1, 2, 3], [4, 5, 6]])
        grad = np.array([[1, 1, 1], [1, 1, 1]])
        return x, grad


class SumTest(unittest.TestCase, FunctionTestMixin):
    do_as_variable = False
    def get_function(self):
        return F.sum

    def get_forward_input_output(self):
        x = as_variable(np.array([[1, 2, 3], [4, 5, 6]])), 0, True
        y = np.array([[5, 7, 9]])
        return x, y

    def get_backward_input_output(self):
        x = as_variable(np.array([[1, 2, 3], [4, 5, 6]])), 0, True
        grad = np.array([[1, 1, 1], [1, 1, 1]])
        return x, grad

    def test_numerical_check(self):
        self.numerical_gradient_check((4, 10))


class MatmulTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return F.matmul

    def get_forward_input_output(self):
        x = np.array([[1, 2, 3], [4, 5, 6]])
        W = np.array([[1, 4], [2, 5], [3, 6]])
        res = np.array([[14, 32], [32, 77]])
        return (x, W), res

    def get_backward_input_output(self):
        x = np.array([[1, 2, 3], [4, 5, 6]])
        W = np.array([[1, 4], [2, 5], [3, 6]])

        grad_x = np.array([[5, 7, 9], [5, 7, 9]])
        grad_W = np.array([[5, 5], [7, 7], [9, 9]])
        return (x, W), (grad_x, grad_W)

    def test_numerical_check(self):
        self.numerical_gradient_check((3, 4), (4, 2))


class MeanSquaredError(unittest.TestCase, ForwardAndBackwardCheckMixin):
    def get_function(self):
        return F.mean_squared_error

    def test_numerical_check(self):
        self.numerical_gradient_check((100, 1), (100, 1))
