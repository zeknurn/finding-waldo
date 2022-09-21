from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA

# Waldo
path = '/home/vic/Downloads/archive/wheres-waldo/Hey-Waldo/64/waldo'
count = 0
for file in listdir(path):
    if count < 1:
        count += 1
        image_path = path + '/' + file

        rgb_image = plt.imread(image_path)
        b, g, r = cv2.split(rgb_image)  # RGB are stored in reverse order in OpenCV
        r_scaled = r / 255
        g_scaled = g / 255
        b_scaled = b / 255

        # Test PCA
        #pca_r = PCA(n_components=None) # None, keeps all components. 32, 48 good range?
        #pca_r.fit(r)

        # Plot it!
        #exp_var = pca_r.explained_variance_ratio_ * 64
        #cum_exp_var = np.cumsum(exp_var)
        # plt.figure(figsize=[7, 10])
        # plt.bar(range(0, 64), exp_var, align='center', label='variance')
        # plt.xlabel('Principal component index')
        # plt.ylabel('Variance percentage')
        # # plt.step(range(0, 64), cum_exp_var, where='mid', label='Cumulative explained variance', color='red')
        # plt.show()

        num = 32
        pca_r = PCA(n_components=num)
        pca_r_trans = pca_r.fit_transform(r_scaled)

        pca_g = PCA(n_components=num)
        pca_g_trans = pca_g.fit_transform(g_scaled)

        pca_b = PCA(n_components=num)
        pca_b_trans = pca_b.fit_transform(b_scaled)

        # For calculating total loss of image quality.
        # print("Explained variances by each channel")
        # print("-----------------------------------")
        # print("Red:", np.sum(pca_r.explained_variance_ratio_) * 100)
        # print("Green:", np.sum(pca_g.explained_variance_ratio_) * 100)
        # print("Blue:", np.sum(pca_b.explained_variance_ratio_) * 100)

        pca_r_original = pca_r.inverse_transform(pca_r_trans)
        pca_g_original = pca_g.inverse_transform(pca_g_trans)
        pca_b_original = pca_b.inverse_transform(pca_b_trans)

        compressed_image = cv2.merge((pca_b_original, pca_g_original, pca_r_original))
        plt.figure(figsize=[2, 2])
        plt.imshow(compressed_image)
        plt.show()
