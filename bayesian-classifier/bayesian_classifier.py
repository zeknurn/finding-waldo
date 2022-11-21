# Naive Bayesian Classifier.
from cmath import e
import os
import numpy as np
import feature_extraction
import pandas as pd

def get_path(filename):
    f = open(filename)
    text = f.read()
    f.close()
    return text

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


def inv_lerp(a: float, b: float, v: float) -> float:
    """Inverse Linar Interpolation, get the fraction between a and b on which v resides.
    Examples
    --------
        0.5 == inv_lerp(0, 100, 50)
        0.8 == inv_lerp(1, 5, 4.2)
    """
    return (v - a) / (b - a)

def calculate_probabilities(path, file, prior_color_waldo, prior_color_notwaldo, is_waldo, total_count):
    #get the current pixel counts for one image
    current_red, current_white, current_skin, current_hair = feature_extraction.extract_color_proportion_single_image(path, file)

    #prior is stored in array order of red, white, skin, hair
    prior_red_waldo = int(prior_color_waldo.iat[0,0])
    prior_white_waldo = int(prior_color_waldo.iat[1,0])
    prior_skin_waldo = int(prior_color_waldo.iat[2,0])
    prior_hair_waldo = int(prior_color_waldo.iat[3,0])

    #get the min and max values of red
    red_min = min(current_red, prior_red_waldo)
    red_max = max(current_red, prior_red_waldo)
    red_inv_lerp = inv_lerp(red_min, red_max, current_red)
    print(red_min, " ", red_max, " ", red_inv_lerp)
    p_waldo = current_red / prior_red_waldo #* p(red|waldo) * p(white|waldo) * p(skin|waldo) * p(hair|waldo)

    prior_red_notwaldo = prior_color_notwaldo.iat[0,0]
    prior_white_notwaldo = prior_color_notwaldo.iat[1,0]
    prior_skin_notwaldo = prior_color_notwaldo.iat[2,0]
    prior_hair_notwaldo = prior_color_notwaldo.iat[3,0]

    p_notwaldo = prior_color_notwaldo.iloc[0,0] #* p(red|not_waldo) * p(white|not_waldo) * p(skin|not_waldo) * p(hair|not_waldo)

    if is_waldo == 1:
        print(total_count, ': Image contains Waldo')
    else:
        print(total_count, ': Image does not contain Waldo')

    if p_waldo > p_notwaldo:
        print(total_count, ': I guess Waldo')
        guess = 1
    else:
        print(total_count, ': I guess not Waldo')
        guess = 0

    if guess == is_waldo:
        score = 1
    else:
        score = 0

    return score


def bayesian_colors(path_waldo, path_notwaldo, prior_color_waldo, prior_color_notwaldo):
        total_count = 0
        total_score = 0

        for file in os.listdir(path_waldo):
            score = calculate_probabilities(path_waldo, file, prior_color_waldo, prior_color_notwaldo, 1, total_count)
            total_score += score
            total_count += 1

        #for file in os.listdir(path_notwaldo):
        #    count, score = calculate_probabilities(path_notwaldo, file, prior_color_waldo, prior_color_notwaldo, 0)
        #    total_count += count
        #    total_score += score

        print('Total score: ', total_score, 'out of ', total_count)
        print(total_score/total_count * 100, '% Accuracy')

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

#with waldo
path_waldo = get_path("path_waldo.txt")

#without waldo
path_notwaldo = get_path("path_notwaldo.txt")

#mixed
path_mixed = get_path("path_mixed.txt")

#for i in range(0, 10):
#    prior_glcm_waldo, prior_glcm_notwaldo = bayesian_glcm(path_mixed, prior_glcm_waldo, prior_glcm_notwaldo)

prior_colors_waldo = pd.read_csv('waldo_color.csv', sep=',', header=None)
prior_colors_notwaldo = pd.read_csv('notwaldo_color.csv', sep=',', header=None)

bayesian_colors(path_waldo, path_notwaldo, prior_colors_waldo, prior_colors_notwaldo)