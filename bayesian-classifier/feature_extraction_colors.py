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

def get_hsv_limits(bgr_value, range_value):
        color = np.uint8([[bgr_value]])

        # convert the color to HSV
        hsvColor = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

        # display the color values
        #print("BGR:", color)
        #print("HSV:", hsvColor)

        # Compute the lower and upper limits
        lowerLimit = (int)(hsvColor[0][0][0]) - range_value, (int)(hsvColor[0][0][1]) - range_value, (int)(hsvColor[0][0][2]) - range_value
        upperLimit = (int)(hsvColor[0][0][0]) + range_value, (int)(hsvColor[0][0][1]) + range_value, (int)(hsvColor[0][0][2]) + range_value

        # display the lower and upper limits
        #print("Lower Limit:",lowerLimit)
        #print("Upper Limit", upperLimit)

        return lowerLimit, upperLimit

def create_mask(hsv, lower, upper, file):
        mask = cv2.inRange(hsv, lower, upper)
                    
        # count non-zero pixels in mask
        count=np.count_nonzero(mask)
        #print('filename:', file,'count:', count)
        return count


def extract_color_proportion_single_image(path, file):
        # load image
        image_path = path + '/' + file
        img = cv2.imread(image_path)

        # convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)

        #waldo red
        #print("red")
        lower, upper = get_hsv_limits([81, 68, 214], 30)
        #create mask
        red_count = create_mask(hsv, lower, upper, file)

        #waldo white
        #print("white")
        lower, upper = get_hsv_limits([250, 255, 237], 30)
        #create mask
        white_count = create_mask(hsv, lower, upper, file)

        #waldo skin
        #print("skin")
        lower, upper = get_hsv_limits([168, 187, 244], 10)
        #create mask
        skin_count = create_mask(hsv, lower, upper, file)

        #waldo hair
        #print("hair")
        lower, upper = get_hsv_limits([54, 37, 40], 30)
        #create mask
        hair_count = create_mask(hsv, lower, upper, file)

        ## save output
        #cv2.imwrite('mask.png', mask)

        ### Display various images to see the steps
        ##cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        ##cv2.resizeWindow('mask', 800, 600)

        #cv2.imshow('mask',mask)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        #print("red: ", red_count, " white: ", white_count, " skin: ", skin_count, " hair: ", hair_count)

        return red_count, white_count, skin_count, hair_count


def extract_color_proportion(path, output_filename):
        red_count = 0
        white_count = 0
        skin_count = 0
        hair_count = 0
        n = 0

        with open(output_filename, 'w') as f:
            for file in listdir(path):
                    n += 1
                    red, white, skin, hair = extract_color_proportion_single_image(path, file)
                    red_count += red
                    white_count += white
                    skin_count += skin
                    hair_count += hair

            average_red = red_count / n
            average_white = white_count / n
            average_skin = skin_count / n
            average_hair = hair_count / n
            print("avg red: ", average_red, " avg white: ", average_white, " avg skin: ", average_skin, " avg hair: ", average_hair) 

            np.savetxt(f, np.array([average_red, average_white, average_skin, average_hair]), delimiter=',')

def get_min_max_colors(path1, output_filename):
        red_min = sys.maxsize
        red_max = 0
        white_min = sys.maxsize
        white_max = 0
        skin_min = sys.maxsize
        skin_max = 0
        hair_min = sys.maxsize
        hair_max = 0
        
        with open(output_filename, 'w') as f:
            for file in listdir(path1):
                red, white, skin, hair = extract_color_proportion_single_image(path1, file)

                if(red < red_min):
                    red_min = red
                elif (red > red_max):
                    red_max = red

                if(white < white_min):
                    white_min = white
                elif (white > white_max):
                    white_max = white

                if(skin < skin_min):
                    skin_min = skin
                elif (skin > skin_max):
                    skin_max = skin

                if(hair < hair_min):
                    hair_min = hair
                elif (hair > hair_max):
                    hair_max = hair
            #print(red_min, " ", red_max)
            np.savetxt(f, np.array([[red_min, red_max], [white_min, white_max], [skin_min, skin_max], [hair_min, hair_max]]), delimiter=',')

#with waldo
path_waldo = get_path("path_waldo.txt")
#extract_color_proportion(path_waldo, "waldo_color.csv")

#without waldo
path_notwaldo = get_path("path_notwaldo.txt")
#extract_color_proportion(path_notwaldo, "notwaldo_color.csv")

#get_min_max_colors(path_waldo, "color_min_max_waldo.csv")
#get_min_max_colors(path_notwaldo, "color_min_max_notwaldo.csv")