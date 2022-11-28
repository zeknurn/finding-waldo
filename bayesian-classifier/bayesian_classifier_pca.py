# example of preparing and making a prediction with a naive bayes model
from sklearn.datasets import make_blobs
from scipy.stats import norm
import numpy as np
from numpy import mean
from numpy import std


def load_waldo_data(nr):
    print("loading data")
    # Read data file and label images with Waldo as 1, and not Waldo as 0.
    with open('features_waldo.csv', 'r') as f:
        x = np.loadtxt(f, delimiter=',', max_rows=nr)
        y = np.ones(shape=(x.shape[0], 1))
    with open('features_notwaldo.csv', 'r') as t:
        x_2 = np.loadtxt(t, delimiter=',', max_rows=nr)
        y_2 = np.zeros(shape=(x_2.shape[0], 1))
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
    print('done')
    return x, y
    # calculate the last row index of the training and testing samples
    # last_row_training = int(rows * training_size_percent / 100)
    # last_row_testing = last_row_training + int(rows * testing_size_percent / 100)
    # slice the matrix into three by using the row indexes
    # x_train = x[:last_row_training]
    # y_train = y[:last_row_training]
    # x_test = x[last_row_training:last_row_testing]
    # y_test = y[last_row_training:last_row_testing]
    # x_valid = x[last_row_testing:]
    # y_valid = y[last_row_testing:]

    # print("sample sizes: data: ", x.shape, " training: ", x_train.shape, " test:", x_test.shape,
    #       " validation:", x_valid.shape)
    # x_train, x_test, y_train, y_test

    # return x_train, x_test, x_valid, y_train, y_test, y_valid


# fit a probability distribution to a univariate data sample


# calculate the independent conditional probability


nr_data_points = 10

# Load data
X_example, y_example = make_blobs(n_samples=50, centers=2, n_features=2, random_state=1)
X, y = load_waldo_data(nr_data_points)
y = np.ndarray.flatten(y)
y = y.astype(int)
print('Waldo shape X: ', X.shape, '/// Example shape X: ', X_example.shape)
print('Waldo shape y: ', y.shape, '/// Example shape y: ', y_example.shape)
print('Waldo dataset:')
print(X[:2], y[:2])
print('Blobs dataset:')
print(X_example[:2], y_example[:2])

# # sort data into classes
Xy0 = X[y == 0]
Xy1 = X[y == 1]
print('Sort data into classes')
print(Xy0.shape, Xy1.shape)


def fit_distribution(data):
    # estimate parameters
    mu = mean(data)
    sigma = std(data)
    # print(mu, sigma)
    # fit distribution
    dist = norm(mu, sigma)
    return dist


def probability(X, prior, dist1, dist2):
    return prior * dist1.pdf(X[0]) * dist2.pdf(X[1])


# If underflow possible, convert multiplication to log.
def probability(X, prior, distributions):
    for i in range(0, 32):
        prior *= distributions[i].pdf(X[i])
        # print('Thinking...', prior, i)

    return prior


# Loop, rad N. 6144
# Create PDFs for y == 0
X1y0 = fit_distribution(Xy0[:, 0])
X2y0 = fit_distribution(Xy0[:, 1])

# Distributions for y == 0
dist0 = np.empty(Xy0.shape[1], dtype=type(X1y0))
for i in range(0, Xy0.shape[1]):
    dist0[i] = fit_distribution(Xy0[:, i])

# Create PDfs for y == 1
X1y1 = fit_distribution(Xy1[:, 0])
X2y1 = fit_distribution(Xy1[:, 1])

# Distributions for y == 1
dist1 = np.empty(Xy1.shape[1], dtype=type(X1y0))
for i in range(0, Xy1.shape[1]):
    dist1[i] = fit_distribution(Xy1[:, i])

priory0 = len(Xy0) / len(X)
priory1 = len(Xy1) / len(X)

score = 0
# classify one example
print(nr_data_points)
for i in range(nr_data_points):
    Xsample, ysample = X[i], y[i]
    # py0 = probability(Xsample, priory0, X1y0, X2y0) # given waldo
    # py1 = probability(Xsample, priory1, X1y1, X2y1) # given not waldo
    py0 = probability(Xsample, priory0, dist0)  # Cumulative probability given not Waldo
    py1 = probability(Xsample, priory1, dist1)  # CDF probability given not Waldo
    print('P(y=0 | %s) = %.3f' % (Xsample, py0 * 100))
    print('P(y=1 | %s) = %.3f' % (Xsample, py1 * 100))
    if py0 > py1 and y[i] == 0:
        score += 1
    elif py1 > py0 and y[i] == 1:
        score += 1
    print('Truth: y=%d' % ysample)

print(score / nr_data_points)

# # generate 2d classification dataset

# print(Xy0.shape, Xy1.shape)
# print(Xy1)
# # calculate priors
# priory0 = len(Xy0) / len(X)
# priory1 = len(Xy1) / len(X)
# # create PDFs for y==0
# distX1y0 = fit_distribution(Xy0[:, 0])
# distX2y0 = fit_distribution(Xy0[:, 1])
# # create PDFs for y==1
# distX1y1 = fit_distribution(Xy1[:, 0])
# distX2y1 = fit_distribution(Xy1[:, 1])

# py0 = probability(Xsample, priory0, distX1y0, distX2y0)
# py1 = probability(Xsample, priory1, distX1y1, distX2y1)
# print('P(y=0 | %s) = %.3f' % (Xsample, py0 * 100))
# print('P(y=1 | %s) = %.3f' % (Xsample, py1 * 100))
# print('Truth: y=%d' % ysample)
