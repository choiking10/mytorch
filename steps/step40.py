import numpy as np
from mytorch import as_variable


def ex1():
    x0 = as_variable(np.array([1, 2, 3]))
    x1 = as_variable(np.array([10]))

    y = x0 + x1
    y.backward()
    print(y)
    print(x0.grad, x1.grad)


def ex2():
    x0 = as_variable(np.array([1, 2, 3]))
    x1 = as_variable(np.array([10]))

    y = x0 * x1
    y.backward()
    print(y)
    print(x0.grad, x1.grad)


def ex3():
    x0 = as_variable(np.array([1, 2, 3]))
    x1 = as_variable(np.array([10]))

    y = x0 - x1
    y.backward()
    print(y)
    print(x0.grad, x1.grad)


def ex4():
    x0 = as_variable(np.array([1, 2, 3]))
    x1 = as_variable(np.array([10]))

    y = x0 / x1
    y.backward()
    print(y)
    print(x0.grad, x1.grad)


ex1()
ex2()
ex3()
ex4()
