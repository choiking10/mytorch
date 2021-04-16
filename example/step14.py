import numpy as np

from mytorch.function import add
from mytorch.variable import Variable


def ex1():
    x = Variable(np.array(3.0))
    y = add(x, x)
    y.backward()
    print(x.grad)

    y = add(add(x, x), x)
    y.backward()
    print(x.grad)  # wrong


def ex2():
    x = Variable(np.array(3.0))
    y = add(x, x)
    y.backward()
    print(x.grad)

    x.zerograd()
    y = add(add(x, x), x)
    y.backward()
    print(x.grad)


ex1()
ex2()
