# Naive Bayesian Classifier.
import os
import feature_extraction
import pandas as pd
import numpy

# Prior probability of seeing waldo, i.e. the ratio of waldo images to non-waldo images in our training set.
# Set arbitrarily to 20%
prior_waldo = 0.2
prior_not_waldo = 1 - prior_waldo

glcm_waldo = 0.9997621193910255
avg_glcm_waldo = 0.0038158859518741435

glcm_notwaldo = 1.007062237607153
avg_glcm_notwaldo = 0.003843748998500584

# Basic first draft of naive bayesian
path = 'C:/Users/Vryds/Desktop/Training/both'
for file in os.listdir(path):
    feature_extraction.gray_scale_cooccurance_single_image(file, 'tmp_out.csv')

    df = pd.read_csv('tmp_out.csv')
    count = numpy.count_nonzero(df)
    glcm_sum = df.to_numpy().sum()

    p_glcm_single = glcm_sum / count


# Probability of containing waldo, i.e. a waldo specific feature, given the image we are looking at contains waldo, i.e:
# P(Glasses | Waldo)

# The times money appear in the spam message, divided by all the words in the spam message.
# the sum glcm value, divided by the count of the glcm values in the waldo image. p(glcm | waldo)
# OR each glcm value, divided by the count of glcm values, think of each glcm value as a word, or feature ->
# p(waldo) * p(glcm1 | waldo) * p(glcm2 | waldo)... etc.

# Probability of the same feature given that the image we are looking at does not contain waldo:
# P(Glasses | Not Waldo)

# Thereafter each we calculate a score for each image.
# A = P(Waldo) * P(Glasses | Waldo) * P(... | Waldo)
# B = P(Not Waldo) * P(Glasses | Not Waldo) * P(... | Not Waldo)
# If A > B, then we classify the image as containing Waldo.

# We first need to split the data into a training, and testing set.

# We then need to look at each image individually and determine the probability of each feature:
# given Waldo, and not Waldo.
