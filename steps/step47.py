import numpy as np

from mytorch import Variable, as_variable
from mytorch.models import MLP
import mytorch.functions as F


def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y


model = MLP((10, 3))
x = Variable(np.array([[0.2, -0.4]]))
y = model(x)
p = F.softmax(y)

p.backward()
print(y)
print(p)

x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])
y = model(x)
loss = F.softmax_cross_entropy(y, t)
print(loss)

loss.backward()
