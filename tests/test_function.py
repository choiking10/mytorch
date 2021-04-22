import unittest

import numpy as np

import mytorch
from mytorch import as_variable, Variable
from mytorch.utils import as_tuple, numerical_gradient
from tests import complex_functions


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

    def forward_check(self, xs, expected_y, using_allclose=False):
        if not isinstance(xs, list) and not isinstance(xs, tuple):
            xs = (xs, )
        xs = list(map(as_variable, xs))
        y = self.forward(*xs)
        if using_allclose:
            self.assertTrue(np.allclose(y.data, expected_y))
        else:
            self.assertEqual(y.data, expected_y)

    def backward_check(self, xs, expected_grads, using_allclose=False):
        xs = list(map(as_variable, as_tuple(xs)))
        expected_grads = as_tuple(expected_grads)
        self.forward_and_backward(*xs)

        if using_allclose:
            for x, expected_grad in zip(xs, expected_grads):
                self.assertTrue(np.allclose(x.grad, expected_grad))
        else:
            for x, expected_grad in zip(xs, expected_grads):
                self.assertEqual(x.grad, expected_grad)

    def test_forward(self):
        self.forward_check(*self.get_forward_input_output())

    def test_backward(self):
        self.backward_check(*self.get_backward_input_output())

    def binary_operator_check(self, f, v1, v2, forward_expect, v1_backward_expect=None, v2_backward_expect=None):
        y = f(v1, v2)
        y.backward()
        self.assertEqual(y.data, forward_expect)
        if v1_backward_expect:
            # if not isinstance(v1_backward_expect, np.ndarray):
            #     v1_backward_expect = np.ndarray(v1_backward_expect)
            self.assertEqual(v1.grad, v1_backward_expect)
            v1.zerograd()
        if v2_backward_expect:
            self.assertEqual(v2.grad, v2_backward_expect)
            v2.zerograd()

    def binary_operator_overloading_check(self, f, v1, v2, forward_expect,
                                          v1_backward_expect=None, v2_backward_expect=None):
        self.binary_operator_check(f, v1, v2, forward_expect, v1_backward_expect, v2_backward_expect)
        self.binary_operator_check(f, v1, v2.data, forward_expect, v1_backward_expect, None)
        self.binary_operator_check(f, v1.data, v2, forward_expect, None, v2_backward_expect)

    def numerical_gradient_check(self, *var_shape_list):
        var_list = [as_variable(np.random.rand(var_shape)) for var_shape in var_shape_list]
        expected_grads = numerical_gradient(self.get_function(), *var_list)
        self.backward_check(var_list, expected_grads, using_allclose=True)


class SquareTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return mytorch.simple_core.square

    def get_forward_input_output(self):
        return np.array(2.0), np.array(4.0)

    def get_backward_input_output(self):
        return np.array(3.0), np.array(6.0)

    def test_numerical_check(self):
        self.numerical_gradient_check(1)


class ExpTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return mytorch.simple_core.exp

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
        return mytorch.simple_core.add

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


class MulTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return mytorch.simple_core.mul

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


class SubTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return mytorch.simple_core.sub

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


class DivTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return mytorch.simple_core.div

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


class PowTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return mytorch.simple_core.pow

    def get_forward_input_output(self):
        x = np.array(2), np.array(3)
        y = np.array(8)
        return x, y

    def get_backward_input_output(self):
        x = np.array(2), np.array(3)
        grad = np.array(12)
        return x, grad


class SinTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return mytorch.simple_core.sin

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
        return mytorch.simple_core.cos

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


class OverloadingTest(unittest.TestCase):
    def test_step20_overloading(self):
        a, b, c = map(as_variable, [3, 4, 5])
        y = (a + b) * c
        self.assertEqual(y.data, np.array(35))
        y.backward()
        self.assertEqual(c.grad, np.array(7))
        self.assertEqual(a.grad, np.array(5))
        self.assertEqual(b.grad, np.array(5))


class MultiPathGraphTest(unittest.TestCase):
    def test_step16_complex_graph(self):
        x = Variable(np.array(2.0))
        a = mytorch.simple_core.square(x)
        y = mytorch.simple_core.add(mytorch.simple_core.square(a), mytorch.simple_core.square(a))
        y.backward()
        self.assertEqual(y.data, np.array(32.0))
        self.assertEqual(x.grad, np.array(64.0))


class NoGradientTest(unittest.TestCase):
    def test_step18_using_no_grad(self):
        with mytorch.simple_core.no_grad():
            x = Variable(np.array(2.0))
            y = mytorch.simple_core.square(x)
            self.assertIsNone(y.creator)

    def test_step18_not_using_no_grad(self):
        x = Variable(np.array(2.0))
        y = mytorch.simple_core.square(x)
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
