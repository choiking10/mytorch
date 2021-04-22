import numpy as np

import mytorch
from mytorch import utils
from mytorch import as_variable

def ex1():
    x = as_variable(np.random.randn(2, 3))
    x.name = 'x'
    print(utils._dot_var(x))
    print(utils._dot_var(x, verbose=True))


def ex2():
    x0 = as_variable(np.random.randn(2, 3))
    x1 = as_variable(np.random.randn(2, 3))
    y = x0 + x1
    print(utils._dot_var(x0), end="")
    print(utils._dot_var(x1), end="")
    print(utils._dot_func(y.creator))


def ex3():
    from tests.complex_functions import goldstein
    x = as_variable(1.0)
    y = as_variable(1.0)
    z = goldstein(x, y)
    z.backward()

    x.name = 'x'
    y.name = 'y'
    z.name = 'z'

    utils.plot_dot_graph(z, verbose=False, to_file='goldstein.png')


ex1()
ex2()
ex3()
