#importing the required libraries
import numpy as np
from skimage.io import imread, imshow
from skimage.filters import prewitt_h,prewitt_v
from skimage.feature import graycomatrix
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from os import listdir
#%matplotlib inline

def get_path(filename):
    f = open(filename)
    text = f.read()
    f.close()
    return text

def prewitt_kernel(path):
        #reading the image 
        image = imread('13_1_1.jpg', as_gray=True)

        #calculating horizontal edges using prewitt kernel
        edges_prewitt_horizontal = prewitt_h(image)
        #calculating vertical edges using prewitt kernel
        edges_prewitt_vertical = prewitt_v(image)

        imshow(edges_prewitt_horizontal, cmap='gray')
        plt.show()

def gray_scale_cooccurance(path, feature_filename):
        with open(feature_filename, 'w') as f:

            glcm = np.empty((256,256,1,1))
            count = 0
            for file in listdir(path):
                count += 1
                image_path = path + '/' + file
                #reading the image 
                image = imread(image_path, as_gray=True)

                glcm += graycomatrix(img_as_ubyte(image), distances=[0], angles=[0], 
                            symmetric=True, normed=True)
            
            average = glcm / count
            #print(glcm)

            np.savetxt(f, average[:, :, 0, 0], delimiter=',')

#with waldo
path = get_path("path_waldo.txt")
gray_scale_cooccurance(path, "probabilities_waldo.csv")

#without waldo
path = get_path("path_notwaldo.txt")
gray_scale_cooccurance(path, "probabilities_notwaldo.csv")


#prewitt_kernel(path)
