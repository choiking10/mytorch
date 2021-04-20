import unittest
from tests.test_function import FunctionTestMixin
from tests import complex_functions


class SphereFunctionTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return complex_functions.sphere

    def get_forward_input_output(self):
        xs = 1, 1
        y = complex_functions.sphere(*xs)
        return xs, y

    def get_backward_input_output(self):
        xs = 1, 1
        grad = 2, 2
        return xs, grad


class MatyasFunctionTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return complex_functions.matyas

    def get_forward_input_output(self):
        xs = 1, 1
        y = self.get_function()(*xs)
        return xs, y

    def get_backward_input_output(self):
        xs = 1, 1
        grad = 0.040000000000000036, 0.040000000000000036
        return xs, grad


class GoldsteinFunctionTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return complex_functions.goldstein

    def get_forward_input_output(self):
        xs = 1, 1
        y = self.get_function()(*xs)
        return xs, y

    def get_backward_input_output(self):
        xs = 1, 1
        grad = -5376.0, 8064.0
        return xs, grad
