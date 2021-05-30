import numpy as np
import matplotlib.pyplot as plt

import mytorch

from mytorch import Variable, Model
from mytorch.models import MLP

import mytorch.functions as F
import mytorch.layers as L


class TwoLayerModel(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = self.l1(x)
        y = F.sigmoid(y)
        y = self.l2(y)
        return y


def main1():
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1) - 0.5
    hidden_size = 10

    model = TwoLayerModel(hidden_size, 1)

    lr = 0.2
    iters = 10000

    for i in range(iters):
        y_pred = model(x)

        model.zerograd()
        loss = F.mean_squared_error(y, y_pred)
        loss.backward()

        for p in model.params():
            p.data -= lr * p.grad.data

        if i % 1000 == 0:
            print(loss)

    with mytorch.no_grad():
        test_x = 0.3
        print(f"sin(2 * {test_x} * pi + 0.3, "
              f"pred: {model(Variable(np.array([test_x])))}, "
              f"expected: {np.sin(2 * test_x * np.pi)}")

        test_x = np.arange(0, 1, 0.01).reshape((100, 1))
        pred_y = model(test_x).data
        gt_y = np.sin(2 * np.pi * test_x)

        plt.scatter(x, y)
        plt.plot(test_x, pred_y, color='r')
        plt.plot(test_x, gt_y, color='g')

        plt.show()


def main2():
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1) - 0.5
    hidden_size = 10

    model = MLP((10, 10, 1))

    lr = 0.2
    iters = 10000

    for i in range(iters):
        y_pred = model(x)

        model.zerograd()
        loss = F.mean_squared_error(y, y_pred)
        loss.backward()

        for p in model.params():
            p.data -= lr * p.grad.data

        if i % 1000 == 0:
            print(loss)

    with mytorch.no_grad():
        test_x = 0.3
        print(f"sin(2 * {test_x} * pi + 0.3, "
              f"pred: {model(Variable(np.array([test_x])))}, "
              f"expected: {np.sin(2 * test_x * np.pi)}")

        test_x = np.arange(0, 1, 0.01).reshape((100, 1))
        pred_y = model(test_x).data
        gt_y = np.sin(2 * np.pi * test_x)

        plt.scatter(x, y)
        plt.plot(test_x, pred_y, color='r')
        plt.plot(test_x, gt_y, color='g')

        plt.show()


main1()
main2()