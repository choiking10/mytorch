import numpy as np
from mytorch import as_variable
from tests.complex_functions import rosenbrock

def ex1():
    x0 = as_variable(0.0)
    x1 = as_variable(2.0)

    y = rosenbrock(x0, x1)
    y.backward()
    print(x0.grad, x1.grad)


def ex2():
    x0 = as_variable(0.0)
    x1 = as_variable(2.0)

    lr = 0.001
    iters = 50000

    for i in range(iters):
        if (i+1) % 1000 == 0:
            print(i+1, x0, x1)

        y = rosenbrock(x0, x1)

        x0.zerograd()
        x1.zerograd()
        y.backward()

        x0.data -= lr * x0.grad
        x1.data -= lr * x1.grad


ex1()
ex2()


