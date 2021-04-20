import numpy as np

import mytorch.simple_core

from mytorch.simple_core import Variable

xs = [Variable(np.array(2)), Variable(np.array(3))]
f = mytorch.simple_core.Add()
ys = f(xs)
y = ys[0]
print(y.data)
