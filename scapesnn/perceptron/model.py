import numpy as np
from enum import Enum


class StepType(Enum):
    HEAVISIDE = 1


class Perceptron:

    def __init__(self, max_iter=1000, n_iter_no_change=5, verbose=False, learning_rate=1, initial_bias=1):
        self.max_iter = max_iter
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.initial_bias = initial_bias

    def step(self, value, type=StepType.HEAVISIDE):
        if type == StepType.HEAVISIDE:
            return 1.0 if (value >= 0) else 0.0
        else:
            return value

    # train the weights for the perceptron using a
    # basic implementation of stochastic gradient descent
    def train(self, features, labels):
        # set weights to zero with additional feature for bias
        w = [0.0 for i in range(features.shape[1]+1)]
        w[0] = self.initial_bias
        w_delta = np.array([w])

        misclassified_ = []
        n_iter = 0
        iter_no_change = 0

        if self.learning_rate == 1 and self.initial_bias == 0:
            return self.train_simple

        for epoch in range(self.max_iter):
            n_iter += 1
            misclassified = 0
            sum_error = 0.0

            for x, label in zip(features, labels):
                prediction = self.predict(x, w)

                error = (label - prediction)
                sum_error += error**2
                w[0] += (self.learning_rate * error)

                if(error):  # misclassified
                    misclassified += 1
                    w[1:] += (self.learning_rate * error * x)
                    w_delta = np.append(w_delta, [w], axis=0)

            misclassified_.append(misclassified)

            if self.verbose:
                print('>epoch=%d, learning rate=%.3f, error=%.3f' %
                      (epoch, self.learning_rate, sum_error))

            if misclassified == 0:
                iter_no_change += 1

            if iter_no_change >= self.n_iter_no_change:
                break

        epochs = [i+1 for i in range(n_iter)]
        return (w, misclassified_, epochs, w_delta)

    # pretty much the same thing as train
    # when learning rate is 1 and initial bias is 0
    # keeping it around to see a different way to do the math
    def train_simple(self, features, labels):

        # set weights to zero with additional
        # feature for bias that always outputs 1
        w = np.zeros(shape=(1, features.shape[1]+1))

        misclassified_ = []
        n_iter = 0
        iter_no_change = 0

        for epoch in range(self.max_iter):
            n_iter += 1
            misclassified = 0
            for x, label in zip(features, labels):
                x = np.insert(x, 0, 1)
                z = np.dot(w, x.transpose())
                target = self.step(z)

                delta = (label - target)

                if(delta):  # misclassified
                    misclassified += 1
                    w += (delta * x)

            misclassified_.append(misclassified)

            if misclassified == 0:
                iter_no_change += 1

            if iter_no_change >= self.n_iter_no_change:
                break

        epochs = np.arange(1, n_iter+1)
        return (w[0], misclassified_, epochs)

    # predict on one or more rows given the weights
    def predict(self, features, weights):
        f_shape = features.shape
        len_f_shape = len(f_shape)
        len_w = len(weights)

        bias = weights[0]

        if len_f_shape == 1 and f_shape[0] == len_w - 1:
            activation = np.dot(weights[1:], features.transpose()) + bias
            return self.step(activation)
        elif len_f_shape == 2 and f_shape[1] == len_w - 1:
            predictions = []
            for i in range(f_shape[0]):
                predictions.append(self.predict(features[i], weights))
            return predictions

    # run predictions on all your data and score your results
    def score(self, features, labels, weights):
        total = len(labels)
        n_correct = 0.0
        predictions = self.predict(features, weights)
        for prediction, label in zip(predictions, labels):
            n_correct += 1.0 if prediction == label else 0.0
            # if self.verbose:
            #     print("Expected=%d, Predicted=%d" % (label, prediction))

        return (n_correct/total)
