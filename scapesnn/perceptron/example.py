from plot2D import plotSVAnimation as plotResults
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron as skPerceptron
import numpy as np
from model import Perceptron


def run():
    iris = load_iris()

    X = iris.data[:, 2:4]  # petal length, petal width
    y = (iris.target == 0).astype(np.int)  # iris-setosa

    percept = Perceptron(learning_rate=1)
    w, misclassified_, epochs, w_d = percept.train(X, y)

    score = percept.score(X, y, w)
    print('Our Perceptron Score:', score)

    clf = skPerceptron()
    clf.fit(X, y)

    score = clf.score(X, y)
    print('SKLearn Perceptron Score:', score)

    plotResults(X, y, w_d)


if __name__ == "__main__":
    run()
