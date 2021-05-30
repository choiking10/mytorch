import numpy as np
import matplotlib.pyplot as plt

import mytorch

from mytorch import Variable
from mytorch.models import MLP
from mytorch.optimizers import MomentumSGD
from mytorch.functions import mean_squared_error


def main():
    # random feed
    np.random.seed(0)

    # dataset
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1) - 0.5

    # hyper parameters
    hidden_size = 10
    lr = 0.2
    iters = 10000

    model = MLP((hidden_size, 1))
    optim = MomentumSGD(model.parameters, lr=lr)

    for i in range(iters):
        pred = model(x)
        model.zerograd()
        loss = mean_squared_error(y, pred)
        loss.backward()
        optim.step()

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


main()
