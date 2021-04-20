import numpy as np
from mytorch.simple_core import Variable

data = np.array(1.0)
x = Variable(data)
print(x.data)

x.data = np.array(2.0)
print(x.data)

###
# supplement
# 다차원배열 = tensor
###

x = np.array(1)
print(x.ndim)  # ndim = number of dimensions

x = np.array([1, 2, 3])
print(x.ndim)

x = np.array([[1, 2, 3], [4, 5, 6]])
print(x.ndim)

