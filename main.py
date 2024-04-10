import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import time
from edges import *
from skimage import io
from skimage.color import rgb2gray

from PIL import Image

def create_pixel_line_art(image_path, pixelation_factor):
    """
    This is the implementation of a canny edge detector on
    an image. Creates a black and white edge detection, 
    i.e. a pixel image.
    """
    original = io.imread(image_path)
    img = rgb2gray(original)

    kernel_size=5
    sigma=1.2
    high=0.03 * pixelation_factor
    low=0.025 * pixelation_factor
    
    kernel = gaussian_kernel(kernel_size, sigma)
    
    smooth_image = conv(img, kernel)
    
    G, theta = gradient(smooth_image)
    
    suppressed_image = non_maximum_suppression(G, theta)
    
    strong_edges, weak_edges = double_thresholding(suppressed_image, high, low)
    
    edges = link_edges(strong_edges, weak_edges)

    pixelated_edges = pixelate_bool(edges, 0.5)

    edges_flipped = np.logical_not(pixelated_edges)

    plt.imshow(edges_flipped, cmap='gray')
    plt.axis('off')
    plt.show()


def pixelate_bool(image_array, scale_down_factor):
    # Convert boolean array to uint8 (255 for True, 0 for False)
    image_uint8 = np.uint8(image_array * 255)

    image = Image.fromarray(image_uint8)  # Convert to PIL image
    original_size = image.size

    # Calculate the new size by scaling down
    new_size = (int(original_size[0] * scale_down_factor), int(original_size[1] * scale_down_factor))

    # Resize down and then back up
    pixelated_image = image.resize(new_size, Image.NEAREST)  # Resize down (pixelate)
    pixelated_image = pixelated_image.resize(original_size, Image.NEAREST)  # Resize up

    # Convert back to a NumPy array and convert to boolean
    pixelated_array = np.asarray(pixelated_image)
    pixelated_bool_array = pixelated_array > 127  # Convert back to boolean

    return pixelated_bool_array

    

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py image_path pixelation_factor")
        sys.exit(1)

    image_path = sys.argv[1]
    pixelation_factor = float(sys.argv[2])
    create_pixel_line_art(image_path, pixelation_factor)