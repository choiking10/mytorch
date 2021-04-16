import numpy as np

from mytorch import function as F
from mytorch.variable import Variable

xs = [Variable(np.array(2)), Variable(np.array(3))]
f = F.Add()
ys = f(xs)
y = ys[0]
print(y.data)
