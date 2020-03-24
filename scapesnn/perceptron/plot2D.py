import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import PyQt5
matplotlib.use('Qt5Agg')


def zero_safe(n):
    epsilon = 0.000000001
    return n if n else epsilon


def line_func(x, w):
    epsilon = 0.000000001
    b, w1, w2 = w
    w1 = zero_safe(w1)
    w2 = zero_safe(w2)
    b = zero_safe(b)

    return (-(b / w2) / (b / w1))*x + (-b / w2)


def plotSV(X, y, w_d):
    plt.axes(xlim=(-1, 8), ylim=(-1, 4))

    plt.scatter(X[:, 0], X[:, 1],
                c=y, cmap=plt.cm.coolwarm)

    cmap = matplotlib.cm.get_cmap('cool', len(w_d))

    x = np.linspace(0, 10, 1000)
    for i in range(len(w_d)):
        plt.plot(x, line_func(x, w_d[i]), color=cmap(i))

    plt.show()


def plotSVAnimation(X, y, w_d):
    x = np.linspace(0, 10, 1000)
    plt.axes(xlim=(-1, 8), ylim=(-1, 4))
    fig, ax = plt.subplots()
    xdata, ydata = [], []

    cmap = matplotlib.cm.get_cmap('cool', len(w_d))

    def init():
        ax.set_xlim(-1, 8)
        ax.set_ylim(-1, 4)
        ax.scatter(X[:, 0], X[:, 1],
                   c=y, cmap=plt.cm.coolwarm)
        return ax,

    def update(frame):
        ax.plot(x, line_func(x, w_d[frame]), color=cmap(frame))
        return ax,

    frames = len(w_d)
    ani = FuncAnimation(fig, update, frames=frames,
                        init_func=init, blit=True)

    plt.show()


def plotEpochs(epochs, misclassified):
    plt.plot(epochs, misclassified)
    plt.xlabel('iterations')
    plt.ylabel('misclassified')
    plt.show()
