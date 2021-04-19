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

# 역전파 x.grad <- A.backward <- a.grad <- B.backward <- b.grad <- C.backward <- y.grad(dy/dy=1)
y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)
