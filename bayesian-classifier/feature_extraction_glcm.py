#importing the required libraries
from asyncio.windows_events import NULL
import numpy as np
from skimage.io import imread, imshow
from skimage.filters import prewitt_h,prewitt_v
from skimage.feature import graycomatrix
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from os import listdir
import cv2
import sys
#%matplotlib inline

def get_path(filename):
    f = open(filename)
    text = f.read()
    f.close()
    return text

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


def gray_scale_cooccurance_single_image(path_to_image, out_file_name):
    with open(out_file_name, 'w') as f:
        # glcm = np.empty((256, 256, 1, 1))
        image = imread(path_to_image, as_gray=True)
        glcm = graycomatrix(img_as_ubyte(image), distances=[0], angles=[0],
                             symmetric=True, normed=True)
        np.savetxt(f, glcm[:, :, 0, 0], delimiter=',')



#with waldo
path_waldo = get_path("path_waldo.txt")
gray_scale_cooccurance(path_waldo, "probabilities_waldo.csv")

#without waldo
path_notwaldo = get_path("path_notwaldo.txt")
gray_scale_cooccurance(path_notwaldo, "probabilities_notwaldo.csv")

# Sum and avg. of non waldo glcm values.
df = pd.read_csv('probabilities_notwaldo.csv')
nr_of_values = numpy.count_nonzero(df)
sum = df.to_numpy().sum()
print('Not Waldo')
print(sum)
print(sum / nr_of_values)

# Sum and avg. of waldo glcm values.
df = pd.read_csv('probabilities_waldo.csv')
nr_of_values = numpy.count_nonzero(df)
sum = df.to_numpy().sum()
print('Waldo')
print(sum)
print(sum / nr_of_values)