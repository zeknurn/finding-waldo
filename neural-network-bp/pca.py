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

        # Convert image to BW using mean.
        image = Image.open(image_path)
        h, w = image.size
        gray = image.convert('L') # Convert to grayscale
        bw = np.asarray(gray).copy()
        bw[bw < 128] = 0
        bw[bw >= 128] = 255

        # For testing
        # bw_image = Image.fromarray(bw)
        # bw_image.show()

        # Converting pixel values to attribute vector
        attributes = []
        for y in range(h):
            pix_row = []
            for x in range(w):
                pix_row.append(bw[x,y])
            attributes.append(pix_row)