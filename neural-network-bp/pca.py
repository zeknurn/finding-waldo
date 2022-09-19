from os import listdir

import numpy as np
import matplotlib as plt
import pandas as pd
from PIL import Image

# Waldo
path = '/home/vic/Downloads/archive/wheres-waldo/Hey-Waldo/64-bw/waldo'
count = 0
for file in listdir(path):
    if count < 1:
        count += 1
        image_path = path + '/' + file
        image = Image.open(image_path)
        pixels = image.load()
        width, height = image.size

        # Testing numpy alternative.

        # Converting each image to a set of attribute vectors.
        attribute_vector = []
        for y in range(height):
            pixels_in_row = []
            for x in range(width):
                pixels_in_row.append(pixels[x, y])
            attribute_vector.append(pixels_in_row)


        Image.fromarray(pixels)



# Not waldo
