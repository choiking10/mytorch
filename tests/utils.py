import numpy as np

from mytorch import as_variable
from mytorch.utils import as_tuple, numerical_gradient


class FunctionTestMixin:
    do_as_variable = True

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
        if self.do_as_variable:
            xs = list(map(as_variable, xs))
        y = self.forward(*xs)
        if using_allclose:
            self.assertAllClose(y.data, expected_y)
        else:
            np.testing.assert_array_equal(y.data, expected_y)

    def backward_check(self, xs, expected_grads, using_allclose=False):
        if self.do_as_variable:
            xs = list(map(as_variable, as_tuple(xs)))
        expected_grads = as_tuple(expected_grads)
        self.forward_and_backward(*xs)

        if using_allclose:
            for x, expected_grad in zip(xs, expected_grads):
                self.assertAllClose(x.grad.data, expected_grad)
        else:
            for x, expected_grad in zip(xs, expected_grads):
                np.testing.assert_array_equal(x.grad.data, expected_grad)

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
            self.assertEqual(v1.grad.data, v1_backward_expect)
            v1.zerograd()
        if v2_backward_expect:
            self.assertEqual(v2.grad.data, v2_backward_expect)
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

    def assertAllClose(self, v, expected):
        np.testing.assert_allclose(v, expected)