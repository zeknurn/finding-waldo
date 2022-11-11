#importing the required libraries
import numpy as np
from skimage.io import imread, imshow
from skimage.filters import prewitt_h,prewitt_v
from skimage.feature import graycomatrix
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
#%matplotlib inline

def prewitt_kernel(path):
        #reading the image 
        image = imread('13_1_1.jpg', as_gray=True)

        #calculating horizontal edges using prewitt kernel
        edges_prewitt_horizontal = prewitt_h(image)
        #calculating vertical edges using prewitt kernel
        edges_prewitt_vertical = prewitt_v(image)

        imshow(edges_prewitt_horizontal, cmap='gray')
        plt.show()

def gray_scale_cooccurance(path):
        #reading the image 
        image = imread('13_1_1.jpg', as_gray=True)
        glcm = graycomatrix(img_as_ubyte(image), distances=[1], angles=[0, np.pi/4, np.pi/2], 
                    symmetric=True, normed=True)

        print(glcm)


def get_path(filename):
    f = open(filename)
    text = f.read()
    f.close()
    return text

path = get_path("path_waldo.txt")
gray_scale_cooccurance(path)
