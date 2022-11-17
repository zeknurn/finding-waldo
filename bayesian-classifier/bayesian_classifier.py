# Naive Bayesian Classifier.

# Prior probability of seeing waldo, i.e. the ratio of waldo images to non-waldo images in our training set.
# Set arbitrarily to 20%
prior_waldo = 0.2
prior_not_waldo = 1 - prior_waldo

# Probability of containing waldo, i.e. a waldo specific feature, given the image we are looking at contains waldo, i.e:
# P(Glasses | Waldo)

# The glcm score of the image, divided by the total avg. gclm score of all waldo images.

# Probability of the same feature given that the image we are looking at does not contain waldo:
# P(Glasses | Not Waldo)

# Thereafter each we calculate a score for each image.
# A = P(Waldo) * P(Glasses | Waldo) * P(... | Waldo)
# B = P(Not Waldo) * P(Glasses | Not Waldo) * P(... | Not Waldo)
# If A > B, then we classify the image as containing Waldo.

# We first need to split the data into a training, and testing set.

# We then need to look at each image individually and determine the probability of each feature:
# given Waldo, and not Waldo.

# Then we calculate the score for that image, and compare our result with the actual answer.
# with open('probabilities_waldo.csv', 'r') as in_waldo:
#     with open('probabilities_notwaldo_preprocessed.csv', 'w') as out_waldo:
#         for line in in_waldo:
#             for value in line:
#                 if value != 0:
#                     out_waldo.write(line)

