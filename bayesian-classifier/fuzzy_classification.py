import copy

import numpy as np
import skfuzzy as fuzz
from os import listdir
import matplotlib.pyplot as plt
import cv2


# Fuzzy classification using Fussy C-Means
# Similar to k-Means but fuzzy. I.e. one cluster can partially belong to other clusters.

def run_fuzzy_classification(path, clusters):
    count = 0
    for file in listdir(path):
        if count != 1:
            # image_path = path + '/' + file
            # image = plt.imread(image_path)
            # X = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # fcm = FCM(n_clusters=2)
            # fcm.fit(X)
            # fcm_centers = fcm.centers
            # fcm_labels = fcm.predict(X)
            #
            # cv2.imshow('Gray image', X)
            # cv2.waitKey(0)


            # https://towardsdatascience.com/image-segmentation-with-clustering-b4bbc98f2ee6
            # Initialize probability matrix randomly - dimensions of image.

            # Calculate the center of clusters, i.e. centoids

            # Calculate new probabilities according to the new center of clusters.

            # Repeat 2 and 3 untill the centers doesn't change.

            count = count + 1


path = "C:/Users/Vryds/Desktop/wheres-waldo/Hey-Waldo/64/waldo"
run_fuzzy_classification(path, 10)
