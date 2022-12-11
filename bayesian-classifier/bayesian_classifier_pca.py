# example of preparing and making a prediction with a naive bayes model
from sklearn.datasets import make_blobs
from scipy.stats import norm
import numpy as np
from numpy import mean
from numpy import std
import time

# loads all waldo data and returns data and label matrices of sample size
def load_waldo_data(sample_size):
    print("loading data")
    # Read data file and label images with Waldo as 1, and not Waldo as 0.
    with open('features_waldo.csv', 'r') as f:
        x = np.loadtxt(f, delimiter=',')
        y = np.ones(shape=(x.shape[0], 1))
    with open('features_notwaldo.csv', 'r') as t:
        x_2 = np.loadtxt(t, delimiter=',')
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
    return x[:sample_size], y[:sample_size], x, y


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
def probability(X, distributions):
    log_arr = np.empty(distributions.shape[0])
    for j in range(0, distributions.shape[0]):
        log_arr[j] = distributions[j].pdf(X[j])
    return log_arr


# performs the log sum trick on the matrices
def logsumexp(x_0, x_1):

    # find whichever of the two matrices has the highest max
    if x_0.max() > x_1.max():
        c = x_0.max()
    else:
        c = x_1.max()

    x_0_log = c + np.log(np.sum(np.exp(x_0 - c)))
    x_1_log = c + np.log(np.sum(np.exp(x_1 - c)))

    return x_0_log, x_1_log

# loads the data and creates the distributions for waldo and not waldo
def init(sample_size):

    # Load data
    X_sample, y_sample, X_all, y_all = load_waldo_data(sample_size)

    y_sample = np.ndarray.flatten(y_sample)
    y_sample = y_sample.astype(int)

    y_all = np.ndarray.flatten(y_all)
    y_all = y_all.astype(int)

    print('Waldo dataset:')
    print('Waldo shape X:', X_sample.shape)
    print('Waldo shape y:', y_sample.shape)

    # sort data into classes
    Xy0 = X_sample[y_sample == 0]
    Xy1 = X_sample[y_sample == 1]

    print('Sorted data into classes')
    print("Not Waldo: ", Xy0.shape, "Waldo: ", Xy1.shape)

    # get distribution type for matrix creation
    X1y0 = fit_distribution(Xy0[:, 0])

    # Distributions for y == 0
    dist0 = np.empty(Xy0.shape[1], dtype=type(X1y0))
    for i in range(0, Xy0.shape[1]):
        dist0[i] = fit_distribution(Xy0[:, i])

    # Distributions for y == 1
    dist1 = np.empty(Xy1.shape[1], dtype=type(X1y0))
    for i in range(0, Xy1.shape[1]):
        dist1[i] = fit_distribution(Xy1[:, i])

    #calculate proportions of waldo and not waldo for prior
    priory0 = len(Xy0) / len(X_all)
    priory1 = len(Xy1) / len(X_all)

    return X_sample, y_sample, dist0, dist1, priory0, priory1


# classify the given samples using the provided distributions and prior data
def classify(X, y, dist0, dist1, priory0, priory1, sample_size):
    accuracy = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    # classify one example
    for i in range(sample_size):

        
        Xsample, ysample = X[i], y[i] # get one data point

        log_arr0 = priory0 * probability(Xsample, dist0)  # Cumulative probability given not Waldo
        log_arr1 = priory1 * probability(Xsample, dist1)  # CDF probability given not Waldo

        py0, py1 = logsumexp(log_arr0, log_arr1)

        # necessary bias relative to the size of the sample. 
        # bigger sample needs a bigger bias. for 100 samples 1 is enough.
        py1 -= 1

        print('Data point: ', i)
        print('P(y=0 | %s)' % py0)
        print('P(y=1 | %s)' % py1)
        if py0 > py1 and y[i] == 0:
            accuracy += 1
            true_negative +=1
        elif py1 < py0 and y[i] == 1:
            false_negative += 1
        elif py1 > py0 and y[i] == 1:
            accuracy += 1
            true_positive += 1
        elif py1 > py0 and y[i] == 0:
            false_positive += 1

        print('Truth: y=%d' % ysample)

    test_waldo_count = np.count_nonzero(y[:] == 1)
    test_notwaldo_count = np.count_nonzero(y[:] == 0)

    # prevent division by zero when using lopsided testing samples.
    if test_waldo_count == 0:
        test_waldo_count = 1

    if test_notwaldo_count == 0:
        test_notwaldo_count = 1

    print("Bayesian classifier without GA")
    print("total accuracy: ", (accuracy / sample_size * 100, "%"))
    print("true_positive: ", (true_positive / test_waldo_count * 100, "%"))
    print("true_negative: ", (true_negative / test_notwaldo_count * 100, "%"))
    print("false_positive: ", (false_positive / test_notwaldo_count * 100, "%"))
    print("false_negative: ", (false_negative / test_waldo_count * 100, "%"))

def main():
    sample_size = 100
    start_time = time.time()

    X, y, dist0, dist1, priory0, priory1 = init(sample_size)
    classify(X, y, dist0, dist1, priory0, priory1, sample_size)

    print("--- %s seconds ---" % (time.time() - start_time))


#main() # disable main if you plan on running the bayesian classifier with GA