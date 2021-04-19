import numpy as np

import mytorch.simple_core

from mytorch.simple_core import Variable

A = mytorch.simple_core.Square()
B = mytorch.simple_core.Exp()
C = mytorch.simple_core.Square()

# y = (e^{x^2})^2
# 계산 그래프의 연결 x -> A -> a -> B -> b -> C -> y
x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

print(y.data)
