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
    epochs = 300
    batch_size = 30
    lr = 0.2
    hidden_size = 100

    train_set = mytorch.datasets.Spiral(train=True)
    test_set = mytorch.datasets.Spiral(train=False)
    train_loader = mytorch.DataLoader(train_set, batch_size)
    test_loader = mytorch.DataLoader(test_set, batch_size)

    train_loss_saver = []
    test_loss_saver = []
    train_acc_saver = []
    test_acc_saver = []

    model = mytorch.models.MLP((hidden_size, 3))
    optim = mytorch.optimizers.MomentumSGD(model.parameters, lr)

    for epoch in range(epochs):

        sum_loss, sum_acc = 0, 0

        for x, t in train_loader:
            pred = model(x)
            loss = mytorch.functions.softmax_cross_entropy(pred, t)
            acc = mytorch.functions.accuracy(pred, t)
            loss.backward()
            optim.step()
            model.zerograd()

            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
        train_loss_saver.append((epoch, sum_loss / len(train_set)))
        train_acc_saver.append((epoch, sum_acc / len(train_set)))
        print(f"[{epoch+1}/{epochs}] train_loss: {sum_loss / len(train_set)}, "
              f"accuracy: {sum_acc / len(train_set)}")

        sum_loss, sum_acc = 0, 0
        with mytorch.no_grad():
            for x, t in test_loader:
                pred = model(x)
                loss = mytorch.functions.softmax_cross_entropy(pred, t)
                acc = mytorch.functions.accuracy(pred, t)
                sum_loss += float(loss.data) * len(t)
                sum_acc += float(acc.data) * len(t)
            test_loss_saver.append((epoch, sum_loss / len(test_set)))
            test_acc_saver.append((epoch, sum_acc / len(train_set)))
            print(f"test_loss: {sum_loss / len(test_set)}, "
                  f"accuracy: {sum_acc / len(test_set)}")


    with mytorch.no_grad():
        xs = None
        pred_classes = None
        for x, t in test_loader:
            pred = model(x)
            pred_class = np.argmax(pred.data, axis=1)
            xs = np.concatenate((xs, x)) if xs is not None else x
            pred_classes = np.concatenate((pred_classes, pred_class)) if pred_classes is not None else pred_class

        plot_decision_regions(xs, pred_classes, model)

    train_loss_saver = np.array(train_loss_saver)
    test_loss_saver = np.array(test_loss_saver)
    plt.plot(train_loss_saver[:, 0], train_loss_saver[:, 1], label='train_loss')
    plt.plot(test_loss_saver[:, 0], test_loss_saver[:, 1], label='test_loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    train_acc_saver = np.array(train_acc_saver)
    test_acc_saver = np.array(test_acc_saver)
    plt.plot(train_acc_saver[:, 0], train_acc_saver[:, 1], label='train_acc')
    plt.plot(test_acc_saver[:, 0], test_acc_saver[:, 1], label='test_acc')
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()


main()
