import numpy as np

from mytorch.simple_core import Variable, Square

x = Variable(np.array(0))
f = Square()
y = f(x)

print(type(y))
print(y.data)
