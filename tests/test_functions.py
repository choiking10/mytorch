import unittest

import numpy as np

import mytorch.functions as F
from tests.utils import FunctionTestMixin


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
