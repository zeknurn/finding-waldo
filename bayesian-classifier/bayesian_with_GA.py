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
def rank_fitness(populations):
    for i in range(0, len(populations)):
        populations[i] = populations[i][0], populations[i][1], cumulative_fitness(populations[i][1])
    populations = sorted(populations, key=lambda x: x[2], reverse=True)
    return populations


def cumulative_fitness(population):
    tmp = 0
    for j in range(0, nr_data_points):
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
        print(i, i + step)
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
        np.random.shuffle(starting_pop)
        tmp = np.copy(starting_pop)
        populations.append((i, tmp, cumulative_fitness(starting_pop)))
    return populations


def run_ga(epochs):
    print('Creating starting population')
    populations = create_population()
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

        populations = rank_fitness(populations)
        print('Rank fitness done - Epoch: ', i)
        print(populations)
    return populations


def init():
    # Load data
    X, y = load_waldo_data(nr_data_points)
    y = np.ndarray.flatten(y)  # Flatten to match dimensions
    y = y.astype(int)
    # sort data into classes
    Xy0 = X[y == 0]
    Xy1 = X[y == 1]
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

    return X, y, dist0, dist1
    # priory0 = len(Xy0) / len(X)
    # priory1 = len(Xy1) / len(X)


# Apply GA #########################
# Representation of distributions.
# The population is represented by an array of indexes, each index is a key for a PDF.
# The GA works by finding a combination of PDFs that yield the highest score.
population_count = 10
nr_data_points = 2
epochs = 1

np.random.seed(1337)  # Reproducible results
X, y, dist0, dist1 = init()
best_fit_population = run_ga(epochs)
# np.savetxt('ga_out_new.csv', best_fit_population, delimiter=',')
print(best_fit_population[0][1])
