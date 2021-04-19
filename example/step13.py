import numpy as np

import mytorch.simple_core

from mytorch.simple_core import Variable


def ex1():
    x0 = Variable(np.array(2.0))
    x1 = Variable(np.array(3.0))
    y = mytorch.simple_core.add(mytorch.simple_core.square(x0), mytorch.simple_core.square(x1))
    y.backward()

    print(y.data)
    print(x0.data)
    print(x1.data)


ex1()


