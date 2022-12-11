from os import listdir
import random
from PIL import Image
  
waldo_overlay_path = r"your\path\to\folder"
notwaldo_path = r"your\path\to\folder"
output_path = r"your\path\to\folder"

def get_image_paths(path):
    image_paths = []
    for file in listdir(path):
        image_path = path + '/' + file
        image_paths.append(image_path)
    return image_paths


def overlay_image(background_image_path, overlay_image_path, i):
    # Opening the primary image (used in background)
    img1 = Image.open(background_image_path)
  
    # Opening the secondary image (overlay image)
    img2 = Image.open(overlay_image_path)
  
    # Pasting img2 image on top of img1 
    # starting at coordinates (0, 0)
    x_pos = random.randint(0, img1.width - img2.width)
    y_pos = random.randint(0, img1.height- img2.height)
    img1.paste(img2, (x_pos, y_pos), mask = img2)
  
    # Displaying the image
    #img1.show()
    # Saving the image
    image_path = output_path + '/' + "waldo_{}.png".format(i+1)
    img1.save(image_path)


waldo_overlay_paths = get_image_paths(waldo_overlay_path)
background_images = get_image_paths(notwaldo_path)

for i in range(len(background_images)):
    waldo_overlay_path = random.choice(waldo_overlay_paths)
    overlay_image(background_images[i], waldo_overlay_path, i)