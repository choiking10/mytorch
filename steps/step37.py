import numpy as np
from mytorch import Variable


def ex1():
    x = Variable(np.array([[1,2,3], [4,5,6]]))
    c = Variable(np.array([[10, 20, 30], [40, 50, 60]]))
    t = x + c
    print(t)


ex1()
