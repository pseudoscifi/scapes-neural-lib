from sklearn.linear_model import Perceptron as skPerceptron
from perceptron import Perceptron
from sklearn.datasets import load_iris
# from matplotlib import pyplot as plt
import numpy as np
# import matplotlib
# matplotlib.use('Qt5Agg')

iris = load_iris()
X = iris.data[:, (2, 3)]  # petal length, petal width
y = (iris.target == 0).astype(np.int)  # iris-setosa

percept = Perceptron(learning_rate=0.1)
w, misclassified_, epochs = percept.train(X, y)

score = percept.score(X, y, w)
print('Our Perceptron Score:', score)

clf = skPerceptron()
clf.fit(X, y)

score = clf.score(X, y)
print('SKLearn Perceptron Score:', score)

# plt.plot(epochs, misclassified_)
# plt.xlabel('iterations')
# plt.ylabel('misclassified')
# plt.show()
