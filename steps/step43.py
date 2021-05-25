import numpy as np
import matplotlib.pyplot as plt
import mytorch
from mytorch import Variable
from mytorch import functions as F


class MyModel:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = Variable(0.01 * np.random.rand(input_dim, hidden_dim))
        self.b1 = Variable(0.01 * np.random.rand(hidden_dim))
        self.W2 = Variable(0.01 * np.random.rand(hidden_dim, output_dim))
        self.b2 = Variable(0.01 * np.random.rand(output_dim))

    def forward(self, x):
        x = F.linear(x, self.W1, self.b1)
        x = F.sigmoid(x)
        x = F.linear(x, self.W2, self.b2)
        # x = F.sigmoid(x)
        return x

    def zerograd(self):
        self.W1.zerograd()
        self.b1.zerograd()
        self.W2.zerograd()
        self.b2.zerograd()

    def update_weight(self, lr=0.01):
        self.W1.data -= self.W1.grad.data * lr
        self.b1.data -= self.b1.grad.data * lr
        self.W2.data -= self.W2.grad.data * lr
        self.b2.data -= self.b2.grad.data * lr


def train():
    x = np.random.rand(100, 1)
    y = np.sin(2 * x * np.pi) + np.random.rand(100, 1)

    model = MyModel(1, 10, 1)
    lr = 0.2
    iters = 20000

    for it in range(iters):
        y_pred = model.forward(x)

        loss = F.mean_squared_error(y_pred, y)
        loss.backward()
        model.update_weight(lr)
        model.zerograd()
        if it % 1000 == 0:
            print(it, loss)

    test_x = 0.3
    print(f"sin(2 * {test_x} * pi + 0.3, "
          f"pred: {model.forward(Variable(np.array([test_x])))}, "
          f"expected: {np.sin(2 * test_x * np.pi) + 0.3}")

    test_x = np.arange(0, 1, 0.01).reshape((100, 1))
    pred_y = model.forward(test_x).data
    # gt_y = np.sin(2 * x * np.pi) + 0.3

    plt.scatter(x, y)
    plt.plot(test_x, pred_y, color='r')

    plt.show()


train()

