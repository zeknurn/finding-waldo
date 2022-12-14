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
import bayesian_classifier_pca
import matplotlib.pyplot as plt
import pandas as pd

# This probability function uses indexes from GA to find which PDF values we should use.
# Works identically as the one above, except different indexes.
# All three parameters have the same shape.
# Matching sample feature and distribution have to be the same across epochs
def probability_ga(Xsample, distributions, population):
    log_arr = np.empty(distributions.shape[0])
    for i in range(0, distributions.shape[0]):
        log_arr[i] = distributions[population[i]].pdf(Xsample[population[i]])
    return log_arr


# calcualtes the fitness score of the given sample using the label and the poplation
def calculate_fitness_score(Xsample, ysample, population, priory0, priory1, dist0, dist1):
    fitness_score = 0
    log_arr0 = priory0 * probability_ga(Xsample, dist0, population)  # Cumulative probability given not Waldo
    log_arr1 = priory1 * probability_ga(Xsample, dist1, population)  # CDF probability given not Waldo

    py0, py1 = bayesian_classifier_pca.logsumexp(log_arr0, log_arr1)

    # necessary bias relative to the size of the sample. 
    # bigger sample needs a bigger bias. for 100 samples 1 is good, for 10 samples 0.1 works
    py1 -= 0.1

    # print('P(y=0 | %s) = %.3f' % (Xsample, py0))
    # print('P(y=1 | %s) = %.3f' % (Xsample, py1))
    # print('Truth: y=%d' % ysample)

    # calculate the fitness by finding on the difference in prediction depending on the truth
    # the bigger the difference the better
    if ysample == 0:
        fitness_score = py0 - py1
    if ysample == 1:
        fitness_score = py1 - py0
    return fitness_score


# Calculates fitness for each population, and replaces the previous score obtained from crossover.
# Then the entire population is sorted, ranked, by fitness score.
def rank_fitness(populations, best_score_current, sample_size, X, y, priory0, priory1, dist0, dist1):
    for i in range(0, len(populations)):
        print("Ranking pop: ", i)
        populations[i] = populations[i][0], populations[i][1], cumulative_fitness(populations[i][1], sample_size, X, y, priory0, priory1, dist0, dist1)
    populations = sorted(populations, key=lambda x: x[2], reverse=True)

    best_score_new = populations[0][2]
    print("Best score: ", best_score_new, "Best current score: ", best_score_current)
    if best_score_new > best_score_current:
        print("Saving new best pop to csv")
        np.savetxt('ga_best_population.csv', populations[0][1], delimiter=',')

    return populations, best_score_new


# calculates cumulative fitness for a population
def cumulative_fitness(population, sample_size, X, y, priory0, priory1, dist0, dist1):
    tmp = 0
    for j in range(0, sample_size):
        print("Calculating fitness for data point nr: ", j)
        tmp += calculate_fitness_score(X[j], y[j], population, priory0, priory1, dist0, dist1)

    return tmp


# performs crossover on two pots by taking the first half from the first pop and combining it with
# the second half of the second pop and vice versa.
def crossover_inner(p1, p2):
    split = int(len(p1) / 2)
    c1 = p1[:split]
    c1 = np.append(c1, p2[split:])
    c2 = p2[:split]
    c2 = np.append(c2, p1[split:])
    return c1, c2

# performs pair wise crossover where the top pop in top half is crossedover with the top pop in bottom half.
# and so on.
def crossover_outer(populations):
    step = int(len(populations) / 2)
    for i in range(0, step):
        p1 = populations[i][1]  # Values
        p2 = populations[i + step][1]
        c1, c2 = crossover_inner(p1, p2)

        # copy index, add new pop sequence and reset fitness score
        populations[i] = populations[i][0], c1, 0 
        # copy index, add new pop sequence and reset fitness score
        populations[i + step] = populations[i + step][0], c2, 0 

    return populations


# Apply mutation
# Selects a population from populations randomly
# Thereafter select both an index and value which is randomly selected and replaced with a random value within range.
def mutate(populations, population_count):
    # gets the indexes of one tenth of the population
    mut_sample_index = np.random.choice(population_count, 10)

    # gets 10 indexes of the location in a pop for mutation
    mut_pop_sequence_index = np.random.choice(6144, 10)

    # gets 10 values that will be placed on the indexes in each pop
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
def create_population(population_count):
    populations = []
    starting_pop = np.arange(6144)
    for i in range(0, population_count):
        print("Creating pop: ", i)
        np.random.shuffle(starting_pop)
        tmp = np.copy(starting_pop)
        populations.append((i, tmp, 0))
    return populations


# runs ga for the given amount of epochs and sample size over the populations
def run_ga(epochs, best_score_current, populations, sample_size, results, X, y, priory0, priory1, dist0, dist1, population_count):

    for i in range(0, epochs):
        print('Epoch: ', i, ':')
        # print('Population: ', populations)
        print('Starting crossover - Epoch: ', i)
        populations = crossover_outer(populations)
        print('Crossover done - Epoch: ', i)
        # print(populations)
        print('Starting mutation - Epoch: ', i)
        populations = mutate(populations, population_count)
        print('Mutation done - Epoch: ', i)
        # print(populations)
        print('Starting rank fitness - Epoch: ', i)

        # rank and get best new fitness score from the batch
        populations, best_score_new = rank_fitness(populations, best_score_current, sample_size, X, y, priory0, priory1, dist0, dist1)

        # save best score for plotting
        new_stats = pd.DataFrame([(best_score_new)], columns=["best_score"])
        results = pd.concat([results, new_stats], ignore_index=True)

        if best_score_new > best_score_current:
            best_score_current = best_score_new

        print('Rank fitness done - Epoch: ', i)
        print(populations)

    # plot the best score over epochs
    plt.clf()
    plt.xlabel("Epochs")
    results.best_score.plot(title="Best Fitness Score")
    plt.savefig("bestscore_pop_size{}_samplesize{}_epochs{}.png".format(len(populations), sample_size, epochs))

# classify the given data 
def classify(X, y, dist0, dist1, priory0, priory1, sample_size, bias):
    print("Loading best pop from csv")
    best_population = np.loadtxt('ga_best_population.csv', delimiter=',').astype(np.int64)

    accuracy = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for i in range(sample_size):
        Xsample, ysample = X[i], y[i]  # one row / picture
        log_arr0 = priory0 * probability_ga(Xsample, dist0, best_population)  # Cumulative probability given not Waldo
        log_arr1 = priory1 * probability_ga(Xsample, dist1, best_population)  # CDF probability given not Waldo

        # Note: Possibly remove prior for GA testing,
        py0, py1 = bayesian_classifier_pca.logsumexp(log_arr0, log_arr1)

        # necessary bias relative to the size of the sample. 
        # bigger sample needs a bigger bias. for 100 samples 1 is enough.
        py1 -= bias

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
    print("total accuracy: ", (accuracy / sample_size * 100, "%"))
    print("true_positive: ", (true_positive / test_waldo_count * 100, "%"))
    print("true_negative: ", (true_negative / test_notwaldo_count * 100, "%"))
    print("false_positive: ", (false_positive / test_notwaldo_count * 100, "%"))
    print("false_negative: ", (false_negative / test_waldo_count * 100, "%"))



# Apply GA #########################
# Representation of distributions.
# The population is represented by an array of indexes, each index is a key for a PDF.
# The GA works by finding a combination of PDFs that yield the highest score.
def main():
    population_count = 2
    sample_size = 4
    epochs = 2

    # necessary bias relative to the size of the sample. 
    # bigger sample needs a bigger bias. for 100 samples 1 is enough. for 10 samples 0.1 is enough
    # set to 0 to try to use GA to remove the need for the bias
    bias = 0

    start_time = time.time()

    # Set random seed for reproducible results 
    np.random.seed(1337)  

    # load data and get dists and priors
    X, y, dist0, dist1, priory0, priory1 = bayesian_classifier_pca.init(sample_size)

    print('Creating starting population')
    populations = create_population(population_count)
    print("Ranking fitness of starting population")

    # create dataframe to store results for plotting
    results = pd.DataFrame(columns=["best_score"])

    # rank and get the best score of the starting pop
    populations, best_score_current = rank_fitness(populations, 0, sample_size, X, y, priory0, priory1, dist0, dist1)

    # add the best score of the starting pop
    new_stats = pd.DataFrame([(best_score_current)], columns=["best_score"])
    results = pd.concat([results, new_stats], ignore_index=True)
    print("Best starting pop score: ", best_score_current)

    # classify using starting pop
    classify(X, y, dist0, dist1, priory0, priory1, sample_size, bias)

    # run ga on the populations
    run_ga(epochs, best_score_current, populations, sample_size, results, X, y, priory0, priory1, dist0, dist1, population_count)

    # classify using the best evolved pop
    classify(X, y, dist0, dist1, priory0, priory1, sample_size, bias)

    print("--- %s seconds ---" % (time.time() - start_time))

main()