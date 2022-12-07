# example of preparing and making a prediction with a naive bayes model
import collections
import copy
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
def load_waldo_data(sample_size):
    print("loading data")
    # Read data file and label images with Waldo as 1, and not Waldo as 0.
    with open('features_waldo.csv', 'r') as f:
        x = np.loadtxt(f, delimiter=',',)
        y = np.ones(shape=(x.shape[0], 1))
    with open('features_notwaldo.csv', 'r') as t:
        x_2 = np.loadtxt(t, delimiter=',',)
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
    return x[:sample_size], y[:sample_size]


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


# This probability function uses indexes from GA to find which PDF values we should use.
# Works identically as the one above, except different indexes.
# All three parameters have the same shape.
# Matching sample feature and distribution have to be the same across epochs
def probability_ga(Xsample, distributions, population):
    log_arr = np.empty(distributions.shape[0])
    for i in range(0, distributions.shape[0]):
        log_arr[i] = distributions[population[i]].pdf(Xsample[population[i]])
    return log_arr


def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))


def calculate_fitness_score(Xsample, ysample, population):
    fitness_score = 0
    log_arr0 = probability_ga(Xsample, dist0, population)  # Cumulative probability given not Waldo
    log_arr1 = probability_ga(Xsample, dist1, population)  # CDF probability given not Waldo

    # Note: Possibly remove prior for GA testing, Add prior here if need be!
    py0 = logsumexp(log_arr0)
    py1 = logsumexp(log_arr1)

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
def rank_fitness(populations, best_score_current):
    for i in range(0, len(populations)):
        print("Ranking pop: ", i)
        populations[i] = populations[i][0], populations[i][1], cumulative_fitness(populations[i][1])
    populations = sorted(populations, key=lambda x: x[2], reverse=True)

    best_score_new = populations[0][2]
    print("Best score: ", best_score_new)
    if best_score_new > best_score_current:
        np.savetxt('ga_best_population.csv', populations[0][1], delimiter=',')

    return populations, best_score_new


def cumulative_fitness(population):
    tmp = 0
    for j in range(0, nr_data_points):
        print("Calculating fitness for data point nr: ", j)
        tmp += calculate_fitness_score(X[j], y[j], population)

    return tmp


def crossover_inner(p1, p2):
    split = int(len(p1) / 2)
    c1 = p1[:split]
    c1 = np.append(c1, p2[split:])
    c2 = p2[:split]
    c2 = np.append(c2, p1[split:])
    return c1, c2


# Crossover populations
# def crossover_outer(populations):
#     for i in range(0, len(populations), 2):
#         p1 = populations[i][1]  # Values
#         p2 = populations[i + 1][1]
#         c1, c2 = crossover_inner(p1, p2)
#         populations[i] = populations[i][0], c1, 0 #populations[i][2]
#         populations[i + 1] = populations[i + 1][0], c2, 0 # populations[i + 1][2]
#     return populations

def crossover_outer(populations):
    step = int(len(populations) / 2)
    for i in range(0, step):
        p1 = populations[i][1]  # Values
        p2 = populations[i + step][1]
        c1, c2 = crossover_inner(p1, p2)
        populations[i] = populations[i][0], c1, 0  # populations[i][2]
        populations[i + step] = populations[i + step][0], c2, 0  # populations[i + 1][2]
    return populations


# Apply mutation
# Selects a population from populations randomly
# Thereafter select both an index and value which is randomly selected and replaced with a random value within range.
def mutate(populations):
    mut_sample_index = np.random.choice(population_count, 10)
    mut_pop_sequence_index = np.random.choice(6144, 10)
    mut_value = np.random.choice(6144, 10)

    for index in mut_sample_index:
        mutation_sample = populations[index][1]
        mutation_sample[mut_pop_sequence_index] = mut_value[random.randint(0, 9)]
    return populations


# Creates ascending numerical array, 0 to N = 6144, which is the number of distributions
# Populations follow this format:
# 0, index
# 1 [123, 234, 35345, 342]
# 2, fitness score
def create_population():
    populations = []
    starting_pop = np.arange(6144)
    for i in range(0, population_count):
        print("Creating pop: ", i)
        np.random.shuffle(starting_pop)
        tmp = np.copy(starting_pop)
        populations.append((i, tmp, 0))
    return populations


def run_ga(epochs):
    print('Creating starting population')
    populations = create_population()
    best_score_current = 0
    for i in range(0, epochs):
        print('Epoch: ', i, ':')
        # print('Population: ', populations)
        print('Starting crossover - Epoch: ', i)
        populations = crossover_outer(populations)
        print('Crossover done - Epoch: ', i)
        # print(populations)
        print('Starting mutation - Epoch: ', i)
        populations = mutate(populations)
        print('Mutation done - Epoch: ', i)
        # print(populations)
        print('Starting rank fitness - Epoch: ', i)

        populations, best_score_new = rank_fitness(populations, best_score_current)

        if best_score_new > best_score_current:
            best_score_current = best_score_new

        print('Rank fitness done - Epoch: ', i)
        print(populations)


def init():
    # Load data
    X, y = load_waldo_data(nr_data_points)
    y = np.ndarray.flatten(y)  # Flatten to match dimensions
    y = y.astype(int)
    print('Waldo dataset:')
    print('Waldo shape X:', X.shape)
    print('Waldo shape y:', y.shape)

    # sort data into classes
    Xy0 = X[y == 0]
    Xy1 = X[y == 1]
    print('Sort data into classes')
    print("Not Waldo: ", Xy0.shape, "Waldo: ", Xy1.shape)
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

    return X, y, dist0, dist1, priory0, priory1


def classify():
    #Load best population
    best_population = np.loadtxt('ga_best_population.csv', delimiter=',').astype(np.int64)

    accuracy = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for i in range(nr_data_points):
        Xsample, ysample = X[i], y[i]  # en rad
        log_arr0 = probability_ga(Xsample, dist0, best_population)  # Cumulative probability given not Waldo
        log_arr1 = probability_ga(Xsample, dist1, best_population)  # CDF probability given not Waldo

        # Test
        py0 = logsumexp(log_arr0) * priory0
        py1 = logsumexp(log_arr1) * priory1

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

    print("Bayesian classifier WITH GA")
    print("total accuracy: ", (accuracy / nr_data_points * 100, "%"))
    print("true_positive: ", (true_positive / test_waldo_count * 100, "%"))
    print("true_negative: ", (true_negative / test_notwaldo_count * 100, "%"))
    print("false_positive: ", (false_positive / test_notwaldo_count * 100, "%"))
    print("false_negative: ", (false_negative / test_waldo_count * 100, "%"))


# Apply GA #########################
# Representation of distributions.
# The population is represented by an array of indexes, each index is a key for a PDF.
# The GA works by finding a combination of PDFs that yield the highest score.
population_count = 100
nr_data_points = 10
epochs = 100
start_time = time.time()

np.random.seed(1337)  # Reproducible results
X, y, dist0, dist1, priory0, priory1 = init()
#run_ga(epochs)

classify()
print("--- %s seconds ---" % (time.time() - start_time))