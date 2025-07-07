import numpy as np
import warnings
from collections import deque

# ============================================================================
# FAST OPTIMIZED FUNCTIONS (RECOMMENDED)
# ============================================================================

def conv_fast(image, kernel):
    """Fast vectorized convolution using numpy."""
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    
    # Pad image
    pad_h, pad_w = Hk // 2, Wk // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    
    # Create output
    out = np.zeros((Hi, Wi))
    
    # Flip kernel once
    kernel = np.flip(kernel)
    
    # Vectorized convolution using sliding windows
    for i in range(Hk):
        for j in range(Wk):
            out += padded[i:i+Hi, j:j+Wi] * kernel[i, j]
    
    return out

def gaussian_kernel_fast(size, sigma):
    """Fast vectorized Gaussian kernel generation."""
    k = size // 2
    
    # Create coordinate grids
    y, x = np.ogrid[-k:k+1, -k:k+1]
    
    # Vectorized computation
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / (2 * np.pi * sigma**2)
    
    return kernel

def partial_x_fast(img):
    """Fast x-derivative using numpy slicing."""
    H, W = img.shape
    out = np.zeros((H, W))
    
    # Use numpy slicing instead of convolution - much faster!
    out[:, 1:-1] = (img[:, 2:] - img[:, :-2]) * 0.5
    # Handle edges
    out[:, 0] = img[:, 1] - img[:, 0]     # left edge
    out[:, -1] = img[:, -1] - img[:, -2]  # right edge
    
    return out

def partial_y_fast(img):
    """Fast y-derivative using numpy slicing."""
    H, W = img.shape
    out = np.zeros((H, W))
    
    # Use numpy slicing instead of convolution - much faster!
    out[1:-1, :] = (img[2:, :] - img[:-2, :]) * 0.5
    # Handle edges  
    out[0, :] = img[1, :] - img[0, :]     # top edge
    out[-1, :] = img[-1, :] - img[-2, :]  # bottom edge
    
    return out

def gradient_fast(img):
    """Fast gradient using optimized partial derivatives."""
    # Use the fast partial derivatives
    filtered_x = partial_x_fast(img)
    filtered_y = partial_y_fast(img)
    
    # Vectorized magnitude and angle computation
    G = np.sqrt(filtered_x**2 + filtered_y**2)
    theta = (np.rad2deg(np.arctan2(filtered_y, filtered_x)) + 180) % 360
    
    return G, theta

def non_maximum_suppression_fast(G, theta):
    """Fast vectorized non-maximum suppression."""
    H, W = G.shape
    
    # Quantize angles to 0, 45, 90, 135
    theta_q = np.round(theta / 45) * 45 % 180
    
    # Create shifted versions of G for all directions
    G_pad = np.pad(G, 1, mode='constant', constant_values=0)
    
    # Direction masks
    mask_0 = (theta_q == 0)      # horizontal
    mask_45 = (theta_q == 45)    # diagonal /
    mask_90 = (theta_q == 90)    # vertical
    mask_135 = (theta_q == 135)  # diagonal \
    
    # Get neighbors for each direction
    neighbors1 = np.zeros_like(G)
    neighbors2 = np.zeros_like(G)
    
    # Horizontal (0°): check left and right
    neighbors1 = np.where(mask_0, G_pad[1:H+1, 0:W], neighbors1)    # left
    neighbors2 = np.where(mask_0, G_pad[1:H+1, 2:W+2], neighbors2)  # right
    
    # Vertical (90°): check up and down  
    neighbors1 = np.where(mask_90, G_pad[0:H, 1:W+1], neighbors1)   # up
    neighbors2 = np.where(mask_90, G_pad[2:H+2, 1:W+1], neighbors2) # down
    
    # Diagonal (45°): check NE and SW
    neighbors1 = np.where(mask_45, G_pad[0:H, 2:W+2], neighbors1)   # NE
    neighbors2 = np.where(mask_45, G_pad[2:H+2, 0:W], neighbors2)   # SW
    
    # Diagonal (135°): check NW and SE
    neighbors1 = np.where(mask_135, G_pad[0:H, 0:W], neighbors1)    # NW
    neighbors2 = np.where(mask_135, G_pad[2:H+2, 2:W+2], neighbors2) # SE
    
    # Suppress non-maxima
    return np.where((G >= neighbors1) & (G >= neighbors2), G, 0)

def double_thresholding(img, high, low):
    """
    Apply double thresholding to edge response.
    
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """
    strong_edges = np.zeros(img.shape, dtype=bool)
    weak_edges = np.zeros(img.shape, dtype=bool)

    strong_edges = (img >= high)
    weak_edges = (img < high) & (img >= low)

    return strong_edges, weak_edges

def link_edges_fast(strong_edges, weak_edges):
    """Fast edge linking using deque and vectorized operations."""
    H, W = strong_edges.shape
    edges = strong_edges.copy()
    weak_edges = weak_edges.copy()
    
    # Use deque for O(1) operations
    queue = deque()
    
    # Find all strong edge pixels at once
    strong_pixels = np.argwhere(strong_edges)
    queue.extend([(y, x) for y, x in strong_pixels])
    
    # Pre-compute neighbor offsets
    neighbor_offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    visited = set()
    
    while queue:
        y, x = queue.popleft()
        
        if (y, x) in visited:
            continue
        visited.add((y, x))
        
        # Check all 8 neighbors
        for dy, dx in neighbor_offsets:
            ny, nx = y + dy, x + dx
            
            # Bounds check and weak edge check
            if (0 <= ny < H and 0 <= nx < W and 
                weak_edges[ny, nx] and not edges[ny, nx]):
                
                edges[ny, nx] = True
                queue.append((ny, nx))
    
    return edges

# ============================================================================
# DEPRECATED FUNCTIONS (BACKWARD COMPATIBILITY ONLY)
# ============================================================================

def conv(image, kernel):
    """
    DEPRECATED: Use conv_fast() instead for better performance.
    
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    warnings.warn(
        "conv() is deprecated and slow. Use conv_fast() instead for 10-100x speedup.",
        DeprecationWarning,
        stacklevel=2
    )
    
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))

    image = np.pad(image, pad_width, mode='edge')

    new_height, new_width = image.shape
    
    kernel = np.flip(kernel)
    
    kernel_width = Wk // 2
    kernel_height = Hk // 2
    
    for x in range(kernel_height, new_height - kernel_height):
        for y in range(kernel_width, new_width - kernel_width):
            neighbourhood = image[x - kernel_height : x + kernel_height + 1, y - kernel_width : y + kernel_width + 1]
            
            out[x - kernel_height, y - kernel_width] = np.sum(np.multiply(neighbourhood, kernel))

    return out

def gaussian_kernel(size, sigma):
    """
    DEPRECATED: Use gaussian_kernel_fast() instead for better performance.
    
    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """
    warnings.warn(
        "gaussian_kernel() is deprecated. Use gaussian_kernel_fast() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    kernel = np.zeros((size, size))

    k = size // 2
    sq_sig = np.square(sigma)
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-(np.square(i - k) + np.square(j - k)) / (2 * sq_sig)) / (2 * np.pi * sq_sig)

    return kernel

def partial_x(img):
    """
    DEPRECATED: Use partial_x_fast() instead for better performance.
    
    Computes partial x-derivative of input img.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """
    warnings.warn(
        "partial_x() is deprecated and slow. Use partial_x_fast() instead for 10x+ speedup.",
        DeprecationWarning,
        stacklevel=2
    )

    x_filter = np.array([[0.5, 0, -0.5]])
    out = conv(img, x_filter)
    return out

def partial_y(img):
    """
    DEPRECATED: Use partial_y_fast() instead for better performance.
    
    Computes partial y-derivative of input img.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """
    warnings.warn(
        "partial_y() is deprecated and slow. Use partial_y_fast() instead for 10x+ speedup.",
        DeprecationWarning,
        stacklevel=2
    )

    y_filter = np.array([[0.5], [0], [-0.5]])
    out = conv(img, y_filter)
    return out

def gradient(img):
    """
    DEPRECATED: Use gradient_fast() instead for better performance.
    
    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).
    """
    warnings.warn(
        "gradient() is deprecated and slow. Use gradient_fast() instead for 50x+ speedup.",
        DeprecationWarning,
        stacklevel=2
    )

    filtered_x = partial_x(img)
    filtered_y = partial_y(img)
    
    G = np.sqrt(filtered_x ** 2 + filtered_y ** 2)
    theta = (np.rad2deg(np.arctan2(filtered_y, filtered_x)) + 180) % 360
    
    return G, theta

def non_maximum_suppression(G, theta):
    """
    DEPRECATED: Use non_maximum_suppression_fast() instead for better performance.
    
    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    warnings.warn(
        "non_maximum_suppression() is deprecated and slow. Use non_maximum_suppression_fast() instead for 20x+ speedup.",
        DeprecationWarning,
        stacklevel=2
    )
    
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    for i in range(H):
        for j in range(W):
            direction = theta[i, j]
            pixel = G[i, j]
            neighbor1 = -1
            neighbor2 = -1
            
            if direction == 45 or direction == 225:
                if i != H - 1 and j != W - 1:
                    neighbor1 = G[i + 1, j + 1]
                if i != 0 and j != 0:
                    neighbor2 = G[i - 1, j - 1]
                
            elif direction == 90 or direction == 270:
                if i != H - 1:
                    neighbor1 = G[i + 1, j]
                if i != 0:
                    neighbor2 = G[i - 1, j]
            
            elif direction == 135 or direction == 315:
                if i != 0 and j != W - 1:
                    neighbor1 = G[i - 1, j + 1]
                if i != H - 1 and j != 0:
                    neighbor2 = G[i + 1, j - 1]
            
            elif direction == 180 or direction == 360 or direction == 0:
                if j != W - 1:
                    neighbor1 = G[i, j + 1]
                if j != 0:
                    neighbor2 = G[i, j - 1]
                
            if G[i, j] >= neighbor1 and G[i, j] >= neighbor2:
                out[i, j] = G[i, j]
            else:
                out[i, j] = 0

    return out

def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """
    DEPRECATED: Use link_edges_fast() instead for better performance.
    
    Find weak edges connected to strong edges and link them.

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """
    warnings.warn(
        "link_edges() is deprecated and slow. Use link_edges_fast() instead for 5x+ speedup.",
        DeprecationWarning,
        stacklevel=2
    )

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)
    
    seen = set()
    queue = []
    
    for i in range(H):
        for j in range(W):
            if edges[i, j]:
                queue.append((i, j))
    
    while queue != []:
        coord = queue.pop(0)
        
        if coord in seen:
            continue
            
        i, j = coord
        
        for n in get_neighbors(i, j, H, W):
            if weak_edges[n[0], n[1]]:
                edges[n[0], n[1]] = True
                queue.append(n)
                
        seen.add(coord)

    return edges