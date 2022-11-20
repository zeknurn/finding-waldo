#importing the required libraries
import numpy as np
from skimage.io import imread, imshow
from skimage.filters import prewitt_h,prewitt_v
from skimage.feature import graycomatrix
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from os import listdir
import cv2
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


def gray_scale_cooccurance_single_image(path_to_image, out_file_name):
    with open(out_file_name, 'w') as f:
        # glcm = np.empty((256, 256, 1, 1))
        image = imread(path_to_image, as_gray=True)
        glcm = graycomatrix(img_as_ubyte(image), distances=[0], angles=[0],
                             symmetric=True, normed=True)
        np.savetxt(f, glcm[:, :, 0, 0], delimiter=',')


def get_hsv_limits(bgr_value, range_value):
        color = np.uint8([[bgr_value]])

        # convert the color to HSV
        hsvColor = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

        # display the color values
        print("BGR:", color)
        print("HSV:", hsvColor)

        # Compute the lower and upper limits
        lowerLimit = (int)(hsvColor[0][0][0]) - range_value, (int)(hsvColor[0][0][1]) - range_value, (int)(hsvColor[0][0][2]) - range_value
        upperLimit = (int)(hsvColor[0][0][0]) + range_value, (int)(hsvColor[0][0][1]) + range_value, (int)(hsvColor[0][0][2]) + range_value

        # display the lower and upper limits
        print("Lower Limit:",lowerLimit)
        print("Upper Limit", upperLimit)

        return lowerLimit, upperLimit

def create_mask(hsv, lower, upper, file):
        mask = cv2.inRange(hsv, lower, upper)
                    
        # count non-zero pixels in mask
        count=np.count_nonzero(mask)
        print('filename:', file,'count:', count)
        return mask


def extract_color_proportion(path, filename):
        with open(filename, 'w') as f:
            for file in listdir(path):
                    # load image
                    image_path = path + '/' + file
                    img = cv2.imread(image_path)

                    # convert to HSV
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    h,s,v = cv2.split(hsv)

                    #waldo red
                    print("red")
                    lower, upper = get_hsv_limits([81, 68, 214], 30)
                    #create mask
                    mask = create_mask(hsv, lower, upper, file)

                    #waldo white
                    print("white")
                    lower, upper = get_hsv_limits([250, 255, 237], 30)
                    #create mask
                    mask = create_mask(hsv, lower, upper, file)

                    #waldo skin
                    print("skin")
                    lower, upper = get_hsv_limits([168, 187, 244], 10)
                    #create mask
                    mask = create_mask(hsv, lower, upper, file)

                    #waldo hair
                    print("hair")
                    lower, upper = get_hsv_limits([54, 37, 40], 30)
                    #create mask
                    mask = create_mask(hsv, lower, upper, file)

                    ## save output
                    #cv2.imwrite('mask.png', mask)

                    ### Display various images to see the steps
                    ##cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
                    ##cv2.resizeWindow('mask', 800, 600)

                    #cv2.imshow('mask',mask)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
                    break


            #np.savetxt(f, glcm[:, :, 0, 0], delimiter=',')



#with waldo
path = get_path("path_waldo.txt")
extract_color_proportion(path, "color.csv")
# gray_scale_cooccurance(path, "probabilities_waldo.csv")
#
# #without waldo
# path = get_path("path_notwaldo.txt")
# gray_scale_cooccurance(path, "probabilities_notwaldo.csv")


#prewitt_kernel(path)
