import numpy as np

from mytorch import function as F
from mytorch.variable import Variable


def ex1():
    # y = (e^{x^2})^2
    # 계산 그래프의 연결 x -> A -> a -> B -> b -> C -> y
    x = Variable(np.array(0.5))
    a = F.square(x)
    b = F.exp(a)
    y = F.square(b)

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)


def ex2():
    x = Variable(np.array(0.5))
    y = F.square(F.exp(F.square(x)))  # 연속 적용
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)


def ex3():
    x = Variable(np.array(0.5))
    y = F.square(F.exp(F.square(x)))  # 연속 적용
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