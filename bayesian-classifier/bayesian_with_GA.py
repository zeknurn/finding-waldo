# example of preparing and making a prediction with a naive bayes model
import collections
import random

import numpy.random
from sklearn.datasets import make_blobs
from scipy.stats import norm
import numpy as np
from numpy import mean
from numpy import std
import time


# Function to load in the dataset, uses PCA values obtained previously.
# Returns X, y, which is the PCA values of the image, and the corresponding true value for that image.
# 1 is Waldo, 0 is not Waldo.
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


# Load data
nr_data_points = 4
X, y = load_waldo_data(nr_data_points)
y = np.ndarray.flatten(y)  # Flatten to match dimensions
y = y.astype(int)

# sort data into classes
Xy0 = X[y == 0]
Xy1 = X[y == 1]


# print('Sort data into classes')
# print(Xy0.shape, Xy1.shape)


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
# def probability(X, prior, distributions):
#     log_arr = np.empty(distributions.shape[0])
#     for j in range(0, distributions.shape[0]):
#         log_arr[j] = distributions[j].pdf(X[j])
#     return log_arr


# This probability function uses indexes from GA to find which PDF values we should use.
# Works identically as the one above, except different indexes.
# All three parameters have the same shape.
def probability_ga(Xsample, distributions, population):
    log_arr = np.empty(distributions.shape[0])
    for k in range(0, distributions.shape[0]):
        log_arr[k] = distributions[population[k]].pdf(Xsample[k])
    return log_arr


def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))


def calculate_fitness_score(Xsample, ysample, population):
    fitness_score = 0
    log_arr0 = probability_ga(Xsample, dist0, population)  # Cumulative probability given not Waldo
    log_arr1 = probability_ga(Xsample, dist1, population)  # CDF probability given not Waldo

    # Note: Possibly remove prior for GA testing
    py0 = priory0 * logsumexp(log_arr0)
    py1 = priory1 * logsumexp(log_arr1)

    # print('P(y=0 | %s) = %.3f' % (Xsample, py0))
    # print('P(y=1 | %s) = %.3f' % (Xsample, py1))
    # print('Truth: y=%d' % ysample)
    if ysample == 0:
        fitness_score = py0 - py1
    if ysample == 1:
        fitness_score = py1 - py0
    return fitness_score


# Calculates fitness for each population, and replaces the previous score obtained from crossover.
# Then the entire population is sorted, ranked, by fitness score.
def rank_fitness():
    print('Rank fitness score:')
    for i in range(0, len(populations)):
        populations[i] = populations[i][0], populations[i][1], cumulative_fitness(populations[i][1])

    print(populations)
    sorted(populations, key=lambda x: x[2])
    print(populations)


def cumulative_fitness(population):
    tmp = 0
    for j in range(0, nr_data_points):
        tmp += calculate_fitness_score(X[j], y[j], population)
    return tmp


def crossover(p1, p2):
    print('Crossover fun')
    split = int(len(p1) / 2)
    c1 = p1[:split]
    c1 = np.append(c1, p2[split:])
    c2 = p2[:split]
    c2 = np.append(c2, p1[split:])
    return c1, c2


# Done for dtype.
X1y0 = fit_distribution(Xy0[:, 0])

# 6144
# One distribution for each colum in X, where y == 0
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

# Apply GA #########################
# Representation of distributions.
# The population is represented by an array of indexes, each index is a key for a PDF.
# The GA works by finding a combination of PDFs that yield the highest score.
population_count = 10
populations = []
# Populations follow this format:
# 0, index
# 1 [123, 234, 35345, 342]
# 2, fitness score
starting_pop = np.arange(6144)  # Creates ascending numerical array, 0 to N = 6144, which is the number of distributions
np.random.seed(1337)  # Reproducible results

print('Initialize starting populations:')
for i in range(0, population_count):
    # np.random.shuffle(starting_pop)
    populations.append((i, starting_pop, cumulative_fitness(starting_pop)))
    print(populations[i])


# Crossover populations
for i in range(0, len(populations), 2):
    p1 = populations[i][1]
    p2 = populations[i + 1][1]
    c1, c2 = crossover(p1, p2)
    populations[i] = populations[i][0], c1, populations[i][2]
    populations[i] = populations[i][0], c2, populations[i][2]

# Rank fitness scores
rank_fitness()

# # Crossover #########################
# print(len(sorted_list))
# for i in range(0, len(sorted_list) - 1, 2):
#     print('Crossover loop')
#     # print('i:', i, ' i + 1: ', i + 1)
#     pop_index1 = sorted_list[i]
#     pop_index2 = sorted_list[i + 1]
#     p1 = populations[pop_index1]
#     p2 = populations[pop_index2]
#     c1, c2 = crossover(p1, p2)
#     # print('C1', c1)
#     # print('C2', c2)
#
#     ## Apply mutation here
#
#     populations[pop_index1] = c1
#     populations[pop_index2] = c2
# print('Crossover Done')
#
# # print('Sorted list:', len(sorted_list))
# # print('Populations: ', len(populations))
#
# sorted_list = list(rank_fitness())
# for key, value in sorted_list:
#     print(key, value)
