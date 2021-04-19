import numpy as np

import mytorch.simple_core

from mytorch.simple_core import Variable


def ex1():
    def f(*x):
        print(x)
    f(1, 2, 3)
    f(1)
    f(1, 2, 3, 4, 5, 6)


def ex2():
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    y = mytorch.simple_core.add(x0, x1)
    print(y.data)


ex1()
ex2()


