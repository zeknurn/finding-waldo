# example of preparing and making a prediction with a naive bayes model
from sklearn.datasets import make_blobs
from scipy.stats import norm
import numpy as np
from numpy import mean
from numpy import std


def load_waldo_data(training_size_percent, testing_size_percent):
    print("loading data")

    with open('features_waldo.csv', 'r') as f:
        x = np.loadtxt(f, delimiter=',')

        y = np.ones(shape=(x.shape[0], 1))
        y = np.append(y, np.zeros([len(x), 1]), axis=1)

    with open('features_notwaldo.csv', 'r') as t:
        x_2 = np.loadtxt(t, delimiter=',')

        y_2 = np.zeros(shape=(x_2.shape[0], 1))
        y_2 = np.append(y_2, np.ones([len(x_2), 1]), axis=1)

    x = np.append(x, x_2, axis=0)
    y = np.append(y, y_2, axis=0)

    # set the random seed to get the same result every run
    np.random.seed(0)

    # get the row count of the matrix
    rows = x.shape[0]

    # shuffle the rows of the matrix
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]

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
    x_train, x_test, y_train, y_test

    return x_train, x_test, x_valid, y_train, y_test, y_valid


# fit a probability distribution to a univariate data sample
def fit_distribution(data):
    # estimate parameters
    mu = mean(data)
    sigma = std(data)
    print(mu, sigma)
    # fit distribution
    dist = norm(mu, sigma)
    return dist


# calculate the independent conditional probability
def probability(X, prior, dist1, dist2):
    return prior * dist1.pdf(X[0]) * dist2.pdf(X[1])


# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# sort data into classes
Xy0 = X[y == 0]
Xy1 = X[y == 1]
# calculate priors
priory0 = len(Xy0) / len(X)
priory1 = len(Xy1) / len(X)
# create PDFs for y==0
distX1y0 = fit_distribution(Xy0[:, 0])
distX2y0 = fit_distribution(Xy0[:, 1])
# create PDFs for y==1
distX1y1 = fit_distribution(Xy1[:, 0])
distX2y1 = fit_distribution(Xy1[:, 1])
# classify one example
Xsample, ysample = X[0], y[0]
py0 = probability(Xsample, priory0, distX1y0, distX2y0)
py1 = probability(Xsample, priory1, distX1y1, distX2y1)
print('P(y=0 | %s) = %.3f' % (Xsample, py0 * 100))
print('P(y=1 | %s) = %.3f' % (Xsample, py1 * 100))
print('Truth: y=%d' % ysample)
