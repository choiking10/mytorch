import numpy as np

from mytorch import function as F
from mytorch.variable import Variable

A = F.Square()
B = F.Exp()
C = F.Square()

# y = (e^{x^2})^2
# 계산 그래프의 연결 x -> A -> a -> B -> b -> C -> y
x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

print(y.data)
