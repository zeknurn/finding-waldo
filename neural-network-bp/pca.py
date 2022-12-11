import csv
from os import listdir
import numpy
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA

# Waldo
def run_PCA(path, feature_filename):
    with open(feature_filename, 'w') as f:
        for file in listdir(path):
            image_path = path + '/' + file
            rgb_image = plt.imread(image_path)

            # split the image rgb channels into three separate matrices
            b, g, r = cv2.split(rgb_image)  # RGB are stored in reverse order in OpenCV

            # normalize the rgb matrices
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


            num = 32 # 32, ca 8% loss, 16 ca 45% loss.

            # perform pca on each of the rgb matrices
            pca_r = PCA(n_components=num)
            pca_r_trans = pca_r.fit_transform(r_scaled)

            pca_g = PCA(n_components=num)
            pca_g_trans = pca_g.fit_transform(g_scaled)

            pca_b = PCA(n_components=num)
            pca_b_trans = pca_b.fit_transform(b_scaled)

            #flatten rgb matrices to single dimension arrays
            feature_vec_r = numpy.asarray(pca_r_trans).flatten()
            feature_vec_g = numpy.asarray(pca_g_trans).flatten()
            feature_vec_b = numpy.asarray(pca_b_trans).flatten()
            
            # add rgb arrays together
            feature_vec = []
            feature_vec = numpy.append(feature_vec, feature_vec_r, 0)
            feature_vec = numpy.append(feature_vec, feature_vec_g, 0)
            feature_vec = numpy.append(feature_vec, feature_vec_b, 0)

            # save array as one row in file
            numpy.savetxt(f, feature_vec.reshape(1, feature_vec.shape[0]), delimiter=',',)

            # For calculating total loss of image quality.
            # print("Explained variances by each channel")
            # print("-----------------------------------")
            # print("Red:", np.sum(pca_r.explained_variance_ratio_) * 100)
            # print("Green:", np.sum(pca_g.explained_variance_ratio_) * 100)
            # print("Blue:", np.sum(pca_b.explained_variance_ratio_) * 100)

            # pca_r_original = pca_r.inverse_transform(pca_r_trans)
            # pca_g_original = pca_g.inverse_transform(pca_g_trans)
            # pca_b_original = pca_b.inverse_transform(pca_b_trans)

            #compressed_image = cv2.merge((pca_b_original, pca_g_original, pca_r_original))
            #plt.figure(figsize=[2, 2])
            #plt.imshow(compressed_image)
            #plt.show()


def get_path(filename):
    f = open(filename)
    text = f.read()
    f.close()
    return text

path = get_path("path_waldo.txt")
run_PCA(path, "features_waldo.csv")

path = get_path("path_notwaldo.txt")
run_PCA(path, "features_notwaldo.csv")