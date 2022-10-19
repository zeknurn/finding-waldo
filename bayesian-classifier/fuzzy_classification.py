import copy

import numpy
from fcmeans import FCM
from os import listdir
import matplotlib.pyplot as plt
import cv2


# Fuzzy classification using Fussy C-Means
# Similar to k-Means but fuzzy. I.e. one cluster can partially belong to other clusters.

def run_fuzzy_classification(path, clusters):
    for file in listdir(path):
        image_path = path + '/' + file
        rgb_image = plt.imread(image_path)
        fcm = FCM(n_cluster=clusters)
        b, g, r = cv2.split(rgb_image)

        fcm.fit(r)
        r_clusters = copy.copy(fcm)
        # g_clusters = copy.deepcopy(fcm.fit(g))
        # b_clusters = copy.deepcopy(fcm.fit(b))

        # result
        # fcm_centers = r_clusters.centers
        # fcm_labels = fcm.predict(r)
        #
        # # plot
        # f, axes = plt.subplots(1, 2, figsize=(11, 5))
        # axes[0].scatter(r[:, 0], r[:, 1], alpha=.1)
        # axes[1].scatter(r[:, 0], r[:, 1], c=fcm_labels, alpha=.1)
        # axes[1].scatter(fcm_centers[:, 0], fcm_centers[:, 1], marker="+", s=500, c='w')
        # plt.savefig('images/basic-clustering-output.jpg')
        # plt.show()


path = "C:/Users/Vryds/Desktop/wheres-waldo/Hey-Waldo/64/waldo"
run_fuzzy_classification(path, 10)
