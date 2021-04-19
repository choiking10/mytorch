import unittest

import numpy as np

import mytorch
from mytorch import as_variable, Variable
from mytorch.utils import as_tuple


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

    def forward_check(self, xs, expected_y):
        xs = as_tuple(as_variable(xs))
        y = self.forward(*xs)
        self.assertEqual(y.data, expected_y)

    def backward_check(self, xs, expected_grads):
        xs = as_tuple(as_variable(xs))
        expected_grads = as_tuple(expected_grads)
        self.forward_and_backward(*xs)
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
        return mytorch.simple_core.square

    def get_forward_input_output(self):
        return np.array(2.0), np.array(4.0)

    def get_backward_input_output(self):
        return np.array(3.0), np.array(6.0)


class ExpTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return mytorch.simple_core.exp

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
        return mytorch.simple_core.add

    def get_forward_input_output(self):
        x = np.array(2), np.array(3)
        y = np.array(5)
        return x, y

    def get_backward_input_output(self):
        x = np.array(2), np.array(3)
        grad_y = np.array(1), np.array(1)
        return x, grad_y

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
        a, b = as_variable((2, 3))
        self.binary_operator_overloading_check(lambda x0, x1: x0+x1, a, b, 5, 1, 1)


class MulTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return mytorch.simple_core.mul

    def get_forward_input_output(self):
        x = np.array(2), np.array(3)
        y = np.array(6)
        return x, y

    def get_backward_input_output(self):
        x = np.array(2), np.array(3)
        grad_y = np.array(3), np.array(2)
        return x, grad_y

    def test_overloading(self):
        a, b = as_variable((2, 3))
        self.binary_operator_overloading_check(lambda x0, x1: x0*x1, a, b, 6, 3, 2)


class SubTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return mytorch.simple_core.sub

    def get_forward_input_output(self):
        x = np.array(2), np.array(3)
        y = np.array(-1)
        return x, y

    def get_backward_input_output(self):
        x = np.array(2), np.array(3)
        grad_y = np.array(1), np.array(-1)
        return x, grad_y

    def test_overloading(self):
        a, b = as_variable((2, 3))
        self.binary_operator_overloading_check(lambda x0, x1: x0-x1, a, b, -1, 1, -1)


class DivTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return mytorch.simple_core.div

    def get_forward_input_output(self):
        x = np.array(8), np.array(2)
        y = np.array(4)
        return x, y

    def get_backward_input_output(self):
        x = np.array(8), np.array(2)
        grad_y = np.array(1/2), np.array(-2)
        return x, grad_y

    def test_overloading(self):
        a, b = as_variable((8, 2))
        self.binary_operator_overloading_check(lambda x0, x1: x0/x1, a, b, 4.0, 1/2, -2)


class PowTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return mytorch.simple_core.pow

    def get_forward_input_output(self):
        x = np.array(2), np.array(3)
        y = np.array(8)
        return x, y

    def get_backward_input_output(self):
        x = np.array(2), np.array(3)
        grad_y = np.array(12)
        return x, grad_y


class OverloadingTest(unittest.TestCase):
    def test_step20_overloading(self):
        a, b, c = as_variable((np.array(3), np.array(4), np.array(5)))
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
