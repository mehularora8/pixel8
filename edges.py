import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    # This variable was 'padded' before. 
    image = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    new_height, new_width = image.shape
    
    kernel = np.flip(kernel)
    
    kernel_width = Wk // 2
    kernel_height = Hk // 2
    
    for x in range(kernel_height, new_height - kernel_height):
        for y in range(kernel_width, new_width - kernel_width):
            neighbourhood = image[x - kernel_height : x + kernel_height + 1, y - kernel_width : y + kernel_width + 1]
            
            out[x - kernel_height, y - kernel_width] = np.sum(np.multiply(neighbourhood, kernel))
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    k = size // 2
    sq_sig = np.square(sigma)
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-(np.square(i - k) + np.square(j - k)) / (2 * sq_sig)) / (2 * np.pi * sq_sig)
    ### END YOUR CODE

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    x_filter = np.array([[0.5, 0, -0.5]])
    
    out = conv(img, x_filter)
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    y_filter = np.array([[0.5], [0], [-0.5]])
    
    out = conv(img, y_filter)
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    filtered_x = partial_x(img)
    filtered_y = partial_y(img)
    
    G = np.sqrt(filtered_x ** 2 + filtered_y ** 2)
    
    theta = (np.rad2deg(np.arctan2(filtered_y, filtered_x)) + 180) % 360
    # Angles are needed clockwise. 
    # Phase of tan is 180
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    #print(G)
    ### BEGIN YOUR CODE
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
                
    ### END YOUR CODE

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=bool)
    weak_edges = np.zeros(img.shape, dtype=bool)

    ### YOUR CODE HERE
    strong_edges = (img >= high)
    weak_edges = (img < high) & (img >= low)
    ### END YOUR CODE

    return strong_edges, weak_edges


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
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)
    
    ### YOUR CODE HERE
    seen = set() # Set of tuples (i, j)
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
    ### END YOUR CODE

    return edges