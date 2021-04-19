import numpy as np

import mytorch.simple_core

from mytorch.simple_core import Variable


def ex1():
    # y = (e^{x^2})^2
    # 계산 그래프의 연결 x -> A -> a -> B -> b -> C -> y
    x = Variable(np.array(0.5))
    a = mytorch.simple_core.square(x)
    b = mytorch.simple_core.exp(a)
    y = mytorch.simple_core.square(b)

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)


def ex2():
    x = Variable(np.array(0.5))
    y = mytorch.simple_core.square(mytorch.simple_core.exp(mytorch.simple_core.square(x)))  # 연속 적용
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)


def ex3():
    x = Variable(np.array(0.5))
    y = mytorch.simple_core.square(mytorch.simple_core.exp(mytorch.simple_core.square(x)))  # 연속 적용
    y.backward()
    print(x.grad)


def ex4():
    x = Variable(np.array(1.0))  # OK
    x = Variable(None)  # OK
    x = Variable(1.0)  #  <class 'float'> is not supported.


ex1()
ex2()
ex3()
ex4()