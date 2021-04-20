import numpy as np

import mytorch.simple_core
import mytorch.utils

from mytorch.simple_core import Variable


def ex1():
    f = mytorch.simple_core.Square()
    x = Variable(np.array(2.0))
    dy = mytorch.utils.numerical_diff(f, x)

    print(dy)


def ex2():
    def f(x):
        A = mytorch.simple_core.Square()
        B = mytorch.simple_core.Exp()
        C = mytorch.simple_core.Square()
        return C(B(A(x)))

    x = Variable(np.array(0.5))
    dy = mytorch.utils.numerical_diff(f, x)
    print(dy)


ex1()
ex2()
