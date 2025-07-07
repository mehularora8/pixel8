#!/usr/bin/env python3
"""
Quick speed test for pixel8 package
"""
import time
import numpy as np
from PIL import Image
import pixel8

def create_test_image(size=300):
    """Create a simple test image"""
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    img[50:150, 50:150] = [0, 0, 0]  # Black square
    img[200:250, 200:250] = [128, 128, 128]  # Gray square
    # Add some lines
    img[100:110, :] = [64, 64, 64]  # Horizontal line
    img[:, 100:110] = [64, 64, 64]  # Vertical line
    
    Image.fromarray(img).save("test_image.png")
    print(f"Created test image: {size}×{size} pixels")
    return "test_image.png"

def test_convolution_speed():
    """Test if convolution is the bottleneck"""
    print("\n" + "="*50)
    print("TESTING CONVOLUTION SPEED")
    print("="*50)
    
    # Test different sizes
    sizes = [100, 200, 300]
    
    for size in sizes:
        img = np.random.rand(size, size)
        kernel = np.random.rand(5, 5)
        
        start = time.time()
        result = pixel8.conv_fast(img, kernel)
        conv_time = time.time() - start
        
        print(f"{size}×{size} image: {conv_time:.2f}s")
        
        if conv_time > 2:
            print("🐌 SLOW - Using nested loops!")
        elif conv_time > 0.5:
            print("🟡 MEDIUM - Could be optimized")
        else:
            print("🚀 FAST - Good performance")

def test_individual_functions():
    """Test individual optimized functions"""
    print("\n" + "="*50)
    print("TESTING INDIVIDUAL FUNCTIONS")
    print("="*50)
    
    # Test with 200x200 image
    img = np.random.rand(200, 200)
    
    # Test gaussian kernel
    start = time.time()
    kernel = pixel8.gaussian_kernel_fast(5, 1.2)
    kernel_time = time.time() - start
    print(f"Gaussian kernel: {kernel_time:.4f}s")
    
    # Test convolution
    start = time.time()
    smoothed = pixel8.conv_fast(img, kernel)
    conv_time = time.time() - start
    print(f"Convolution: {conv_time:.4f}s")
    
    # Test gradient
    start = time.time()
    G, theta = pixel8.gradient(smoothed)
    gradient_time = time.time() - start
    print(f"Gradient: {gradient_time:.4f}s")
    
    # Test non-max suppression
    start = time.time()
    suppressed = pixel8.non_maximum_suppression_fast(G, theta)
    nms_time = time.time() - start
    print(f"Non-max suppression: {nms_time:.4f}s")
    
    # Test double thresholding
    start = time.time()
    strong, weak = pixel8.double_thresholding(suppressed, 0.045, 0.0375)
    threshold_time = time.time() - start
    print(f"Double thresholding: {threshold_time:.4f}s")
    
    # Test edge linking
    start = time.time()
    edges = pixel8.link_edges_fast(strong, weak)
    linking_time = time.time() - start
    print(f"Edge linking: {linking_time:.4f}s")

def test_full_pipeline():
    """Test the full pixel art pipeline"""
    print("\n" + "="*50)
    print("TESTING FULL PIPELINE")
    print("="*50)
    
    # Create test image
    image_path = create_test_image(300)
    
    print("Running pixel art generation...")
    start_total = time.time()
    
    # Run with timing
    result = pixel8.create_pixel_line_art(
        image_path, 
        pixelation_factor=1.5, 
        show_result=False
    )
    
    total_time = time.time() - start_total
    
    print(f"\nResults:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Output size: {result.shape}")
    print(f"Pixels processed: {result.size:,}")
    print(f"Speed: {result.size / total_time:.0f} pixels/second")
    
    # Performance assessment
    if total_time > 10:
        print("🐌 VERY SLOW - Major optimization needed")
    elif total_time > 3:
        print("🟡 SLOW - Some optimization recommended")
    else:
        print("🚀 GOOD - Acceptable performance")

def main():
    print("PIXEL8 SPEED TEST")
    print("="*50)
    
    try:
        # Check what functions are available
        print("Available functions:")
        print(pixel8.__all__)
        
        # Test individual functions
        test_individual_functions()
        
        # Test convolution specifically
        test_convolution_speed()
        
        # Test full pipeline
        test_full_pipeline()
        
        print("\n" + "="*50)
        print("Speed test complete!")
        print("If you see 🐌 SLOW, we need to optimize that function.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Make sure pixel8 is installed: pip install -e .")

if __name__ == "__main__":
    main() 