# example of preparing and making a prediction with a naive bayes model
from sklearn.datasets import make_blobs
from scipy.stats import norm
import numpy as np
from numpy import mean
from numpy import std
import time


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


nr_data_points = 2
# Load data
X, y = load_waldo_data(nr_data_points)
y = np.ndarray.flatten(y)
y = y.astype(int)
print('Waldo dataset:')
print('Waldo shape X:', X.shape)
print('Waldo shape y:', y.shape)

# # sort data into classes
Xy0 = X[y == 0]
Xy1 = X[y == 1]
print('Sort data into classes')
print(Xy0.shape, Xy1.shape)


# This function fits each and every single variable in a column to a normal distribution.
# The normal distribution for each variable is then stored in an array upon returning.
def fit_distribution(data):
    mu = mean(data)
    sigma = std(data)
    dist = norm(mu, sigma)
    return dist


# The Bayesian probability function.
# Here we use the values obtained from the probability density function of each variable.
# Since values can range from -2 sigma, +2 sigma, probability values either over-, or underflow.
# We use the log-sum trick to normalize values
# P(x1 | W) * P(x2 | W)...
def probability(X, prior, distributions):
    log_arr = np.empty(distributions.shape[0])
    for j in range(0, distributions.shape[0]):
        log_arr[j] = distributions[j].pdf(X[j])
    return log_arr


def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))


start_time = time.time()
# Done for dtype.
X1y0 = fit_distribution(Xy0[:, 0])

# Distributions for y == 0
dist0 = np.empty(Xy0.shape[1], dtype=type(X1y0))
for i in range(0, Xy0.shape[1]):
    dist0[i] = fit_distribution(Xy0[:, i])

# Distributions for y == 1
dist1 = np.empty(Xy1.shape[1], dtype=type(X1y0))
for i in range(0, Xy1.shape[1]):
    dist1[i] = fit_distribution(Xy1[:, i])

priory0 = len(Xy0) / len(X)
priory1 = len(Xy1) / len(X)

# Apply GA
# Representation of distributions
# A fitness function
# recombination
# mutation

score = 0
# classify one example
# print(nr_data_points)
# for i in range(nr_data_points):
#     Xsample, ysample = X[i], y[i]  # en rad
#     # py0 = probability(Xsample, priory0, X1y0, X2y0) # given not Waldo
#     # py1 = probability(Xsample, priory1, X1y1, X2y1) # given Waldo
#     log_arr0 = probability(Xsample, priory0, dist0)  # Cumulative probability given not Waldo
#     log_arr1 = probability(Xsample, priory1, dist1)  # CDF probability given not Waldo
#
#     # Test
#     py0 = priory0 * logsumexp(log_arr0)
#     py1 = priory1 * logsumexp(log_arr1)
#
#     print('P(y=0 | %s) = %.3f' % (Xsample, py0 * 100))
#     print('P(y=1 | %s) = %.3f' % (Xsample, py1 * 100))
#     if py0 > py1 and y[i] == 0:
#         score += 1
#     elif py1 > py0 and y[i] == 1:
#         score += 1
#     print('Truth: y=%d' % ysample)
#
# print("--- %s seconds ---" % (time.time() - start_time))
# print(score / nr_data_points)
