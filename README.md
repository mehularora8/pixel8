# pixel8

Generate pixel art from images using Canny edge detection.

## Overview

pixel8 is a Python package that transforms regular images into pixel art using the Canny edge detection algorithm. It's perfect for creating stylized artwork, tattoo designs, or just having fun with image processing.

## Installation

```bash
pip install pixel8
```

## Usage

### Command Line Interface

```bash
# Basic usage
pixel8 input.jpg 1.5

# Save output to file
pixel8 input.jpg 2.0 --output pixel_art.png

# Don't display the result
pixel8 input.jpg 1.5 --output result.png --no-show
```

### Python API

```python
import pixel8

# Generate pixel art
result = pixel8.create_pixel_line_art(
    image_path="input.jpg",
    pixelation_factor=1.5,
    output_path="output.png",
    show_result=True
)

# Use individual components
from pixel8 import gaussian_kernel, conv, gradient

# Create custom edge detection pipeline
kernel = gaussian_kernel(5, 1.2)
# ... rest of your pipeline
```

## Parameters

- **pixelation_factor** (float): Controls the pixelation threshold. Higher values = fewer pixels, more stylized result

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- scikit-image
- Pillow

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/pixel8.git
cd pixel8

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e .[dev]
```

## Motivation

I stumbled across some beautiful pixel art and wanted to teach myself how to make it. I am also in the process of deciding what tattoo to get next. Thought I'd put the two together and design my next tattoo in pixel art.

## License

MIT License - see LICENSE file for details.
