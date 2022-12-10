# Naive Bayesian Classifier based on color averages
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


def inv_lerp(a: float, b: float, v: float) -> float:
    """Inverse Linar Interpolation, get the fraction between a and b on which v resides.
    Examples
    --------
        0.5 == inv_lerp(0, 100, 50)
        0.8 == inv_lerp(1, 5, 4.2)
    """
    return (1 - ((v - a) / (b - a))) + 0.001


def calculate_probabilities(path, file, prior_color_waldo, prior_color_notwaldo, is_waldo, total_count
                            , min_max_color_waldo, min_max_color_notwaldo):
    #get the current pixel counts for one image
    current_red, current_white, current_skin, current_hair = feature_extraction.extract_color_proportion_single_image(path, file)

    #prior is stored in array order of red, white, skin, hair
    prior_avg_red_waldo = int(prior_color_waldo.iat[0,0])
    prior_avg_white_waldo = int(prior_color_waldo.iat[1,0])
    prior_avg_skin_waldo = int(prior_color_waldo.iat[2,0])
    prior_avg_hair_waldo = int(prior_color_waldo.iat[3,0])

    #get inverse lerp of each color containing waldo
    if current_red > prior_avg_red_waldo:
        red_inv_lerp_waldo = inv_lerp(prior_avg_red_waldo, min_max_color_waldo.iat[0,1], current_red)
    else:
        red_inv_lerp_waldo = inv_lerp(prior_avg_red_waldo, min_max_color_waldo.iat[0,0], current_red)

    if current_white > prior_avg_white_waldo:
        white_inv_lerp_waldo = inv_lerp(prior_avg_white_waldo, min_max_color_waldo.iat[1,1], current_white)
    else:
        white_inv_lerp_waldo = inv_lerp(prior_avg_white_waldo, min_max_color_waldo.iat[1,0], current_white)

    if current_skin > prior_avg_skin_waldo:
        skin_inv_lerp_waldo = inv_lerp(prior_avg_skin_waldo, min_max_color_waldo.iat[2,1], current_skin)
    else:
        skin_inv_lerp_waldo = inv_lerp(prior_avg_skin_waldo, min_max_color_waldo.iat[2,0], current_skin)

    if current_hair > prior_avg_hair_waldo:
        hair_inv_lerp_waldo = inv_lerp(prior_avg_hair_waldo, min_max_color_waldo.iat[3,1], current_hair)
    else:
        hair_inv_lerp_waldo = inv_lerp(prior_avg_hair_waldo, min_max_color_waldo.iat[3,0], current_hair)

    #print("waldo. min: ", min_max_color_waldo.iat[0,0], "max ",  min_max_color_waldo.iat[0,1], " avg: ", prior_avg_red_waldo, " current_red: ", current_red, " inv lerp: ", red_inv_lerp_waldo)
    #print("waldo. min: ", min_max_color_waldo.iat[1,0], "max ",  min_max_color_waldo.iat[1,1], " avg: ", prior_avg_white_waldo,  " current_white: ", current_white, " inv lerp: ", white_inv_lerp_waldo)
    #print("waldo. min: ", min_max_color_waldo.iat[2,0], "max ",  min_max_color_waldo.iat[2,1], " avg: ", prior_avg_skin_waldo,  " current_skin: ", current_skin, " inv lerp: ", skin_inv_lerp_waldo)
    #print("waldo. min: ", min_max_color_waldo.iat[3,0], "max ",  min_max_color_waldo.iat[3,1], " avg: ", prior_avg_hair_waldo,  " current_skin: ", current_hair, " inv lerp: ", hair_inv_lerp_waldo)
    
    p_waldo = red_inv_lerp_waldo * white_inv_lerp_waldo * skin_inv_lerp_waldo * hair_inv_lerp_waldo

    prior_avg_red_notwaldo = prior_color_notwaldo.iat[0,0]
    prior_avg_white_notwaldo = prior_color_notwaldo.iat[1,0]
    prior_avg_skin_notwaldo = prior_color_notwaldo.iat[2,0]
    prior_avg_hair_notwaldo = prior_color_notwaldo.iat[3,0]

    #get inverse lerp of each color NOT containing waldo
    if current_red > prior_avg_red_notwaldo:
        red_inv_lerp_notwaldo = inv_lerp(prior_avg_red_notwaldo, min_max_color_notwaldo.iat[0,1], current_red)
    else:
        red_inv_lerp_notwaldo = inv_lerp(prior_avg_red_notwaldo, min_max_color_notwaldo.iat[0,0], current_red)

    if current_white > prior_avg_white_notwaldo:
        white_inv_lerp_notwaldo = inv_lerp(prior_avg_white_notwaldo, min_max_color_notwaldo.iat[1,1], current_white)
    else:
        white_inv_lerp_notwaldo = inv_lerp(prior_avg_white_notwaldo, min_max_color_notwaldo.iat[1,0], current_white)

    if current_skin > prior_avg_skin_notwaldo:
        skin_inv_lerp_notwaldo = inv_lerp(prior_avg_skin_notwaldo, min_max_color_notwaldo.iat[2,1], current_skin)
    else:
        skin_inv_lerp_notwaldo = inv_lerp(prior_avg_skin_notwaldo, min_max_color_notwaldo.iat[2,0], current_skin)

    if current_hair > prior_avg_hair_notwaldo:
        hair_inv_lerp_notwaldo = inv_lerp(prior_avg_hair_notwaldo, min_max_color_notwaldo.iat[3,1], current_hair)
    else:
        hair_inv_lerp_notwaldo = inv_lerp(prior_avg_hair_notwaldo, min_max_color_notwaldo.iat[3,0], current_hair)

    #print("notwaldo. min: ", min_max_color_notwaldo.iat[0,0], "max ",  min_max_color_notwaldo.iat[0,1], " avg: ", prior_avg_red_notwaldo, " current_red: ", current_red, " inv lerp: ", red_inv_lerp_notwaldo)
    #print("notwaldo. min: ", min_max_color_notwaldo.iat[1,0], "max ",  min_max_color_notwaldo.iat[1,1], " avg: ", prior_avg_white_notwaldo, " current_white: ", current_white, " inv lerp: ", white_inv_lerp_notwaldo)
    #print("notwaldo. min: ", min_max_color_notwaldo.iat[2,0], "max ",  min_max_color_notwaldo.iat[2,1], " avg: ", prior_avg_skin_notwaldo,  " current_skin: ", current_skin, " inv lerp: ", skin_inv_lerp_notwaldo)
    #print("notwaldo. min: ", min_max_color_notwaldo.iat[3,0], "max ",  min_max_color_notwaldo.iat[3,1], " avg: ", prior_avg_hair_notwaldo,  " current_hair: ", current_hair, " inv lerp: ", hair_inv_lerp_notwaldo)

    p_notwaldo = red_inv_lerp_notwaldo * white_inv_lerp_notwaldo * skin_inv_lerp_notwaldo * hair_inv_lerp_notwaldo

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


def bayesian_colors(path_waldo, path_notwaldo, prior_color_waldo, prior_color_notwaldo,
                   min_max_color_waldo, min_max_color_notwaldo):
        total_count = 0
        total_score = 0

        for file in os.listdir(path_waldo):
            score = calculate_probabilities(path_waldo, file, prior_color_waldo, prior_color_notwaldo, 1,
                                           total_count, min_max_color_waldo, min_max_color_notwaldo)
            total_score += score
            total_count += 1

        for file in os.listdir(path_notwaldo):
            score = calculate_probabilities(path_notwaldo, file, prior_color_waldo, prior_color_notwaldo, 0,
                                           total_count, min_max_color_waldo, min_max_color_notwaldo)
            total_score += score
            total_count += 1

        print('Total score: ', total_score, 'out of ', total_count)
        print(total_score/total_count * 100, '% Accuracy')


#with waldo
path_waldo = get_path("path_waldo.txt")

#without waldo
path_notwaldo = get_path("path_notwaldo.txt")

#mixed
path_mixed = get_path("path_mixed.txt")

prior_colors_waldo = pd.read_csv('waldo_color.csv', sep=',', header=None)
prior_colors_notwaldo = pd.read_csv('notwaldo_color.csv', sep=',', header=None)

min_max_colors_waldo = pd.read_csv('color_min_max_waldo.csv', sep=',', header=None)
min_max_colors_notwaldo = pd.read_csv('color_min_max_notwaldo.csv', sep=',', header=None)

bayesian_colors(path_waldo, path_notwaldo, prior_colors_waldo, prior_colors_notwaldo, min_max_colors_waldo, min_max_colors_notwaldo)