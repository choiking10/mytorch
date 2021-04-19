import unittest

import numpy as np

import mytorch
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


class MulTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return F.mul

    def get_forward_input_output(self):
        x = np.array(2), np.array(3)
        y = np.array(6)
        return x, y

    def get_backward_input_output(self):
        x = np.array(2), np.array(3)
        grad_y = np.array(3), np.array(2)
        return x, grad_y


class SubTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return F.sub

    def get_forward_input_output(self):
        x = np.array(2), np.array(3)
        y = np.array(-1)
        return x, y

    def get_backward_input_output(self):
        x = np.array(2), np.array(3)
        grad_y = np.array(1), np.array(-1)
        return x, grad_y


class DivTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return F.div

    def get_forward_input_output(self):
        x = np.array(8), np.array(2)
        y = np.array(4)
        return x, y

    def get_backward_input_output(self):
        x = np.array(8), np.array(2)
        grad_y = np.array(1/2), np.array(-2)
        return x, grad_y


class PowTest(unittest.TestCase, FunctionTestMixin):
    def get_function(self):
        return F.pow

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

    def test_step21_overloading(self):
        a = Variable(np.array(3))
        y = 5.0 * (a + 4.0)
        self.assertEqual(y.data, np.array(35))
        y.backward()
        self.assertEqual(a.grad, np.array(5))


class MultiPathGraphTest(unittest.TestCase):
    def test_step16_complex_graph(self):
        x = Variable(np.array(2.0))
        a = F.square(x)
        y = F.add(F.square(a), F.square(a))
        y.backward()
        self.assertEqual(y.data, np.array(32.0))
        self.assertEqual(x.grad, np.array(64.0))


class NoGradientTest(unittest.TestCase):
    def test_step18_using_no_grad(self):
        with mytorch.no_grad():
            x = Variable(np.array(2.0))
            y = F.square(x)
            self.assertIsNone(y.creator)

    def test_step18_not_using_no_grad(self):
        x = Variable(np.array(2.0))
        y = F.square(x)
        self.assertIsNotNone(y.creator)
