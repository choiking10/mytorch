import numpy as np

from mytorch import function as F
from mytorch.variable import Variable


def ex1():
    f = F.Square()
    x = Variable(np.array(2.0))
    dy = F.numerical_diff(f, x)

    print(dy)


def ex2():
    def f(x):
        A = F.Square()
        B = F.Exp()
        C = F.Square()
        return C(B(A(x)))

    x = Variable(np.array(0.5))
    dy = F.numerical_diff(f, x)
    print(dy)


ex1()
ex2()
