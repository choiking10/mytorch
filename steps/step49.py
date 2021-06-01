import math

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

import numpy as np
import mytorch


def show_spiral_data(x, t):
    color = ['r', 'g', 'b']
    marker = ['o', 'x', '^']

    for c in range(3):
        p_class = np.where([t == c])
        plt.scatter(x[p_class, 0], x[p_class, 1], color=color[c], marker=marker[c])

    plt.show()


def plot_decision_regions(x, t, model, resolution=0.02):
    # 마커와 컬러맵 설정
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(t))])

    # 결정 경계 그리기
    x1_min, x1_max = -1, 1
    x2_min, x2_max = -1, 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = model(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = np.argmax(Z.data, axis=1).reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for c in range(3):
        p_class = np.where([t == c])
        plt.scatter(x[p_class, 0], x[p_class, 1], color=colors[c], marker=markers[c])

    plt.show()


def main():
    train_set = mytorch.datasets.Spiral()

    epochs = 300
    batch_size = 30
    data_size = len(train_set)
    max_iter = math.ceil(data_size / batch_size)

    lr = 0.2
    hidden_size = 100
    loss_saver = []

    model = mytorch.models.MLP((hidden_size, 3))
    optim = mytorch.optimizers.MomentumSGD(model.parameters, lr)

    for epoch in range(epochs):

        index = np.random.permutation(data_size)
        sum_loss = 0

        for i in range(max_iter):
            batch_index = index[i * batch_size: (i+1) * batch_size]
            batch = [train_set[i] for i in batch_index]
            batch_x = np.array([example[0] for example in batch])
            batch_t = np.array([example[1] for example in batch])

            pred = model(batch_x)
            loss = mytorch.functions.softmax_cross_entropy(pred, batch_t)
            loss.backward()
            optim.step()
            model.zerograd()

            sum_loss += float(loss.data) * len(batch_t)
        loss_saver.append((epoch, sum_loss / data_size))
        print(f"[{epoch+1}/{epochs}] loss: {sum_loss / data_size}")

    val_x, val_t = mytorch.datasets.get_spiral(False)
    with mytorch.no_grad():
        pred = model(val_x)
        pred_class = np.argmax(pred.data, axis=1)

        # show_spiral_data(val_x, val_t)
        plot_decision_regions(val_x, pred_class, model)

    loss_saver = np.array(loss_saver)
    plt.plot(loss_saver[:, 0], loss_saver[:, 1])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()


main()
