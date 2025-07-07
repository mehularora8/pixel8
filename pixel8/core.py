"""
Core pixel art generation functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from PIL import Image

from .edges import (
    gaussian_kernel_fast, conv_fast, gradient_fast, non_maximum_suppression_fast, 
    double_thresholding, link_edges_fast
)


def create_pixel_line_art(image_path, pixelation_factor, output_path=None, show_result=True):
    """
    Create pixel line art from an image using Canny edge detection.
    
    Args:
        image_path (str): Path to the input image
        pixelation_factor (float): Control for pixelation threshold. Higher = fewer pixels
        output_path (str, optional): Path to save the output image. If None, image is not saved.
        show_result (bool): Whether to display the result using matplotlib
        
    Returns:
        numpy.ndarray: The processed pixel art as a boolean array
    """
    # Read and convert image to grayscale
    original = io.imread(image_path)
    img = rgb2gray(original)

    # Edge detection parameters
    kernel_size = 5
    sigma = 1.2
    high = 0.03 * pixelation_factor
    low = 0.025 * pixelation_factor
    
    # Apply Canny edge detection using optimized functions
    kernel = gaussian_kernel_fast(kernel_size, sigma)
    smooth_image = conv_fast(img, kernel)
    G, theta = gradient_fast(smooth_image)  # Use ultra-fast gradient!
    suppressed_image = non_maximum_suppression_fast(G, theta)
    strong_edges, weak_edges = double_thresholding(suppressed_image, high, low)
    edges = link_edges_fast(strong_edges, weak_edges)

    # Apply pixelation
    pixelated_edges = pixelate_bool(edges, 0.5)
    
    # Invert for better visualization (black lines on white background)
    edges_flipped = np.logical_not(pixelated_edges)

    # Display result
    if show_result:
        plt.figure(figsize=(10, 8))
        plt.imshow(edges_flipped, cmap='gray')
        plt.axis('off')
        plt.title('Pixel Line Art')
        plt.show()
    
    # Save result if output path is provided
    if output_path:
        save_pixel_art(edges_flipped, output_path)
    
    return edges_flipped


def pixelate_bool(image_array, scale_down_factor):
    """
    Apply pixelation effect to a boolean image array.
    
    Args:
        image_array (numpy.ndarray): Boolean array representing the image
        scale_down_factor (float): Factor to scale down the image for pixelation
        
    Returns:
        numpy.ndarray: Pixelated boolean array
    """
    # Convert boolean array to uint8 (255 for True, 0 for False)
    image_uint8 = np.uint8(image_array * 255)
    
    image = Image.fromarray(image_uint8)
    original_size = image.size

    # Calculate the new size by scaling down
    new_size = (
        int(original_size[0] * scale_down_factor), 
        int(original_size[1] * scale_down_factor)
    )

    # Resize down and then back up for pixelation effect
    pixelated_image = image.resize(new_size, Image.NEAREST)
    pixelated_image = pixelated_image.resize(original_size, Image.NEAREST)

    # Convert back to boolean array
    pixelated_array = np.asarray(pixelated_image)
    pixelated_bool_array = pixelated_array > 127

    return pixelated_bool_array


def save_pixel_art(image_array, output_path):
    """
    Save pixel art to file.
    
    Args:
        image_array (numpy.ndarray): Boolean array representing the pixel art
        output_path (str): Path to save the output image
    """
    # Convert boolean to uint8
    image_uint8 = np.uint8(image_array * 255)
    
    # Save using PIL
    image = Image.fromarray(image_uint8)
    image.save(output_path) 