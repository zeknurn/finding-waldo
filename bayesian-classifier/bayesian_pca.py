import numpy as np
import pandas as pd
import sklearn.utils
# Sklearn Bayesian
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


def load_data(training_size_percent, testing_size_percent):
    print("loading data")

    with open('features_waldo.csv', 'r') as f:
        x = np.loadtxt(f, delimiter=',')

        y = np.ones(shape=(x.shape[0], 1))
        # y = np.append(y, np.zeros([len(x), 1]), axis=1)

    with open('features_notwaldo.csv', 'r') as t:
        x_2 = np.loadtxt(t, delimiter=',', max_rows=None)

        y_2 = np.zeros(shape=(x_2.shape[0], 1))
        # y_2 = np.append(y_2, np.ones([len(x_2), 1]), axis=1)

    x = np.append(x, x_2, axis=0)
    y = np.append(y, y_2, axis=0)

    # set the random seed to get the same result every run
    np.random.seed(2)

    # get the row count of the matrix
    rows = x.shape[0]

    # shuffle the rows of the matrix
    # np.random.shuffle(x)
    x, y = sklearn.utils.shuffle(x, y)

    # calculate the last row index of the training and testing samples
    last_row_training = int(rows * training_size_percent / 100)
    last_row_testing = last_row_training + int(rows * testing_size_percent / 100)

    # slice the matrix into three by using the row indexes
    x_train = x[:last_row_training]
    y_train = y[:last_row_training]
    x_test = x[last_row_training:last_row_testing]
    y_test = y[last_row_training:last_row_testing]
    x_valid = x[last_row_testing:]
    y_valid = y[last_row_testing:]

    print("sample sizes: data: ", x.shape, " training: ", x_train.shape, " test:", x_test.shape,
          " validation:", x_valid.shape)
    # x_train, x_test, y_train, y_test

    return x_train, x_test, x_valid, y_train, y_test, y_valid


X_train, X_test, X_valid, y_train, y_test, y_valid = load_data(training_size_percent=80, testing_size_percent=20)
gnb = GaussianNB()
bnb = BernoulliNB()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)

y_pred = gnb.fit(X_train, y_train).predict(X_test)
y_pred_bernouli = bnb.fit(X_train, y_train).predict(X_test)

# print('/////////// TEST')
# print(y_test)
# print('//////////// PREDICTION')
# print(y_pred)

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred_bernouli).sum()))