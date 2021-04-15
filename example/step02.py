import numpy as np

from mytorch.function import Square
from mytorch.variable import Variable

x = Variable(np.array(0))
f = Square()
y = f(x)

print(type(y))
print(y.data)
