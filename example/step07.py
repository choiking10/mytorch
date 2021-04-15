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

assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x


y.grad = np.array(1.0)
y.backward()
print(x.grad)
