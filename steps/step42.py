import numpy as np
import mytorch
import mytorch.functions as F


class LinearRegression:
    def __init__(self, input_dim):
        self.W = mytorch.Variable(np.zeros(input_dim))
        self.b = mytorch.Variable(np.zeros(input_dim))

    def predict(self, x):
        return self.W * x + self.b

    def zerograd(self):
        self.W.zerograd()
        self.b.zerograd()

    def update_weight(self, lr):
        self.W.data -= lr * self.W.grad.data
        self.b.data -= lr * self.b.grad.data


def mean_square_error(predict, Y):
     return F.sum((predict-Y) ** 2) / len(predict)


def linear_regression():
    x = np.random.rand(1000, 1)
    y = 5 + x * 2 + np.random.rand(1000, 1)

    model = LinearRegression((1, 1))

    lr = 0.25
    iters  = 10000

    for i in range(iters):

        model.zerograd()
        pred = model.predict(x)
        loss = mean_square_error(pred, y)
        loss.backward()
        model.update_weight(lr)

        if i % 100 == 0:
            print(i, model.W, model.b, loss)

    print(f"{10} * 2 + 5 = {model.predict(10).data[0][0]}")


linear_regression()
