import cv2
import numpy as np

def preprocess_and_detect(image, method='canny'):
    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    if method == 'canny':
        # Canny edge detection
        edges = cv2.Canny(blurred, 100, 200)

    elif method == 'sobel':
        # Sobel edge detection (calculating both x and y gradients)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)  # X gradient
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)  # Y gradient
        edges = np.sqrt(sobelx ** 2 + sobely ** 2)  # Magnitude of gradient
        edges = np.uint8(np.absolute(edges))  # Convert to uint8 format

        # Apply binary threshold to reduce weak edges and noise
        _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

    elif method == 'roberts':
        # Roberts edge detection (using Roberts cross kernels)
        roberts_cross_v = np.array([[1, 0], [0, -1]], dtype=np.float32)
        roberts_cross_h = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        vertical = cv2.filter2D(blurred, -1, roberts_cross_v)
        horizontal = cv2.filter2D(blurred, -1, roberts_cross_h)
        edges = np.sqrt(vertical ** 2 + horizontal ** 2)  # Magnitude of gradient
        edges = np.uint8(np.absolute(edges))  # Convert to uint8 format

        # Apply binary threshold to reduce weak edges and noise
        _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

    return edges