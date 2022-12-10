# Naive Bayesian Classifier based on gray-level co-occurance matrix
from cmath import e
import os
import numpy as np
import feature_extraction
import pandas as pd


def bayesian_glcm(path, prior_glcm_waldo, prior_glcm_notwaldo):
    count = 0
    score = 0
    for file in os.listdir(path):

        feature_extraction.gray_scale_cooccurance_single_image(path + '/' + file, 'tmp_out.csv')
        current_glcm = pd.read_csv('tmp_out.csv', sep=',', header=None)
        # current_glcm.replace(0, 1)

        # print('Current: ', current_glcm.shape)
        # print('Prior: ', prior_glcm_waldo.shape)

        score_a = np.dot(current_glcm, prior_glcm_waldo)
        score_b = np.dot(current_glcm, prior_glcm_notwaldo)
        prior_glcm_waldo = score_a
        prior_glcm_notwaldo = score_b

        tmp_a = 1.0
        for j in range(0, score_a.shape[0]):
            for i in range(0, score_a.shape[1]):
                if score_a[i, j] != 0:
                    tmp_tmp = tmp_a
                    if tmp_tmp * score_a[i, j] != 0:
                        tmp_a *= score_a[i, j]
                    else:
                        break

        tmp_b = 1.0
        for j in range(0, score_b.shape[0]):
            for i in range(0, score_b.shape[1]):
                if score_b[i, j] != 0:
                    tmp_tmp = tmp_b
                    if tmp_tmp * score_b[i, j] != 0:
                        tmp_b *= score_b[i, j]
                    else:
                        break

        true = 1
        guess = 0
        #24, both, 39 both1
        #if count >= 39:
        #    print('True image does not contain Waldo')
        #    true = 0
        #else:
        #    print('True image contains Waldo')
        #    true = 1

        if tmp_a > tmp_b:
            print(count, ': I guess Waldo')
            guess = 1
        else:
            print(count, ': I guess not Waldo')
            guess = 0

        if guess == true:
            score += 1

        count += 1
        print('Score A:', tmp_a)
        print('Score B:', tmp_b)

    print('Total score: ', score, 'out of ', count)
    print(score/count, '% Accuracy')
    return prior_glcm_waldo, prior_glcm_notwaldo


# Prior probability of seeing waldo, i.e. the ratio of waldo images to non-waldo images in our training set.
# Set arbitrarily to 20%

# glcm_waldo = 0.9997621193910255
# avg_glcm_waldo = 0.0038158859518741435
#
# glcm_notwaldo = 1.007062237607153
# avg_glcm_notwaldo = 0.003843748998500584

# Basic first draft of naive bayesian
prior_waldo = 0.2
prior_not_waldo = 1 - prior_waldo

prior_glcm_waldo = pd.read_csv('probabilities_waldo.csv', sep=',', header=None)
prior_glcm_notwaldo = pd.read_csv('probabilities_notwaldo.csv', sep=',', header=None)
# prior_glcm_waldo = prior_glcm_waldo.replace(0.0, 1.0)

for i in range(0, 10):
    prior_glcm_waldo, prior_glcm_notwaldo = bayesian_glcm(path_mixed, prior_glcm_waldo, prior_glcm_notwaldo)
