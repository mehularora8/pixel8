"""
pixel8 - Generate pixel art from images.
"""

__version__ = "1.0.0"
__author__ = "Mehul Arora"
__email__ = "aroramehul8@gmail.com"

from .core import create_pixel_line_art, pixelate_bool
from .edges import (
    conv_fast, gaussian_kernel_fast, partial_x_fast, partial_y_fast, 
    gradient_fast, non_maximum_suppression_fast, 
    double_thresholding, link_edges_fast
)

# Also import old functions for backward compatibility
from .edges import (
    conv, gaussian_kernel, partial_x, partial_y, gradient,
    non_maximum_suppression, link_edges
)

__all__ = [
    "create_pixel_line_art",
    "pixelate_bool",
    # Fast functions (recommended)
    "conv_fast",
    "gaussian_kernel_fast", 
    "partial_x_fast",
    "partial_y_fast",    
    "gradient_fast",
    "non_maximum_suppression_fast",
    "double_thresholding",
    "link_edges_fast",
    # Deprecated functions (backward compatibility)
    "conv",
    "gaussian_kernel",
    "partial_x", 
    "partial_y",
    "gradient",
    "non_maximum_suppression",
    "link_edges"
] 