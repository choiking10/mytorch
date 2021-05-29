import numpy as np
import matplotlib.pyplot as plt

from mytorch import Variable
import mytorch.functions as F
import mytorch.layers as L


np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1) - 0.5


l1 = L.Linear(10)
l2 = L.Linear(1)


def predict(inputs):
    o = l1(inputs)
    o = F.sigmoid(o)
    o = l2(o)
    return o


lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.zerograd()
    l2.zerograd()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print(loss)


test_x = 0.3
print(f"sin(2 * {test_x} * pi + 0.3, "
      f"pred: {predict(Variable(np.array([test_x])))}, "
      f"expected: {np.sin(2 * test_x * np.pi)}")

test_x = np.arange(0, 1, 0.01).reshape((100, 1))
pred_y = predict(test_x).data
gt_y = np.sin(2 * np.pi * test_x)

plt.scatter(x, y)
plt.plot(test_x, pred_y, color='r')
plt.plot(test_x, gt_y, color='g')

plt.show()
