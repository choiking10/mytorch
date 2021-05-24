import unittest

import numpy as np

import mytorch
from mytorch import as_variable, Variable

from tests import complex_functions
from tests.utils import FunctionTestMixin


class SquareTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return mytorch.core.square

    def get_forward_input_output(self):
        return np.array(2.0), np.array(4.0)

    def get_backward_input_output(self):
        return np.array(3.0), np.array(6.0)

    def test_numerical_check(self):
        self.numerical_gradient_check(1)


class ExpTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return mytorch.core.exp

    def get_forward_input_output(self):
        x = np.array(2.0)
        y = np.array(np.exp(x))
        return x, y

    def get_backward_input_output(self):
        x = np.array(3.0)
        grad = np.array(np.exp(x))
        return x, grad

    def test_numerical_check(self):
        self.numerical_gradient_check(1)


class AddTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return mytorch.core.add

    def get_forward_input_output(self):
        x = np.array(2), np.array(3)
        y = np.array(5)
        return x, y

    def get_backward_input_output(self):
        x = np.array(2), np.array(3)
        grad = np.array(1), np.array(1)
        return x, grad

    def test_step14_forward_same_variable(self):
        x = Variable(np.array(3.0))
        xs = (x, x)
        expected_ys = np.array(6.0)
        self.forward_check(xs, expected_ys)

    def test_step14_backward_same_variable(self):
        x = Variable(np.array(3.0))
        xs = (x, x)
        expected_grad = np.array(2.0)
        self.backward_check(xs, expected_grad)

    def test_overloading(self):
        a, b = map(as_variable, (2, 3))
        self.binary_operator_overloading_check(lambda x0, x1: x0+x1, a, b, 5, 1, 1)

    def test_gradient_check(self):
        self.numerical_gradient_check(1, 1)

    def test_gradient_broadcast_check(self):
        self.numerical_gradient_check((2, 3), (2, 1))


class MulTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return mytorch.core.mul

    def get_forward_input_output(self):
        x = np.array(2), np.array(3)
        y = np.array(6)
        return x, y

    def get_backward_input_output(self):
        x = np.array(2), np.array(3)
        grad = np.array(3), np.array(2)
        return x, grad

    def test_overloading(self):
        a, b = map(as_variable, (2, 3))
        self.binary_operator_overloading_check(lambda x0, x1: x0*x1, a, b, 6, 3, 2)

    def test_numerical_check(self):
        self.numerical_gradient_check(1, 1)

    def test_gradient_broadcast_check(self):
        self.numerical_gradient_check((2, 3), (2, 1))


class SubTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return mytorch.core.sub

    def get_forward_input_output(self):
        x = np.array(2), np.array(3)
        y = np.array(-1)
        return x, y

    def get_backward_input_output(self):
        x = np.array(2), np.array(3)
        grad = np.array(1), np.array(-1)
        return x, grad

    def test_overloading(self):
        a, b =map(as_variable, (2, 3))
        self.binary_operator_overloading_check(lambda x0, x1: x0-x1, a, b, -1, 1, -1)

    def test_numerical_check(self):
        self.numerical_gradient_check(1, 1)

    def test_gradient_broadcast_check(self):
        self.numerical_gradient_check((2, 3), (2, 1))


class DivTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return mytorch.core.div

    def get_forward_input_output(self):
        x = np.array(8), np.array(2)
        y = np.array(4)
        return x, y

    def get_backward_input_output(self):
        x = np.array(8), np.array(2)
        grad = np.array(1/2), np.array(-2)
        return x, grad

    def test_overloading(self):
        a, b = map(as_variable, (8, 2))
        self.binary_operator_overloading_check(lambda x0, x1: x0/x1, a, b, 4.0, 1/2, -2)

    def test_numerical_check(self):
        self.numerical_gradient_check(1, 1)

    def test_gradient_broadcast_check(self):
        self.numerical_gradient_check((2, 3), (2, 1))


class PowTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return mytorch.core.pow

    def get_forward_input_output(self):
        x = np.array(2), np.array(3)
        y = np.array(8)
        return x, y

    def get_backward_input_output(self):
        x = np.array(2), np.array(3)
        grad = np.array(12)
        return x, grad


class OverloadingTest(unittest.TestCase):
    def test_step20_overloading(self):
        a, b, c = map(as_variable, [3, 4, 5])
        y = (a + b) * c
        self.assertEqual(y.data, np.array(35))
        y.backward()
        self.assertEqual(c.grad.data, np.array(7))
        self.assertEqual(a.grad.data, np.array(5))
        self.assertEqual(b.grad.data, np.array(5))


class MultiPathGraphTest(unittest.TestCase):
    def test_step16_complex_graph(self):
        x = Variable(np.array(2.0))
        a = mytorch.core.square(x)
        y = mytorch.core.add(mytorch.core.square(a), mytorch.core.square(a))
        y.backward()
        self.assertEqual(y.data, np.array(32.0))
        self.assertEqual(x.grad.data, np.array(64.0))


class NoGradientTest(unittest.TestCase):
    def test_step18_using_no_grad(self):
        with mytorch.core.no_grad():
            x = Variable(np.array(2.0))
            y = mytorch.core.square(x)
            self.assertIsNone(y.creator)

    def test_step18_not_using_no_grad(self):
        x = Variable(np.array(2.0))
        y = mytorch.core.square(x)
        self.assertIsNotNone(y.creator)


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
