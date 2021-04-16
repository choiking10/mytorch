import numpy as np

from mytorch import function as F
from mytorch.variable import Variable


def ex1():
    x0 = Variable(np.array(2.0))
    x1 = Variable(np.array(3.0))
    y = F.add(F.square(x0), F.square(x1))
    y.backward()

    print(y.data)
    print(x0.data)
    print(x1.data)


ex1()


