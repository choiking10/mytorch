import contextlib

import numpy as np

import mytorch
import mytorch.simple_core
from mytorch.simple_core import Variable, square, add


def ex1():
    x0 = Variable(np.array(2.0))
    x1 = Variable(np.array(3.0))
    t = add(x0, x1)
    y = add(x0, t)
    y.backward()

    print(y.grad, t.grad)
    print(x0.grad, x1.grad)


def ex2():
    @contextlib.contextmanager
    def config_test():
        print('start')  # 전처리
        try:
            yield
        finally:
            print('done')
    with config_test():
        print('process...')


def ex3():
    with mytorch.simple_core.no_grad():
        x = Variable(np.array(2.0))
        y = square(x)
        print("no_grad generation", y.creator)

    x = Variable(np.array(2.0))
    y = square(x)
    print("grad generation", y.creator)


ex1()
ex2()
ex3()
