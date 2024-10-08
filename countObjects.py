import cv2

def count_objects(edges):
    # Find contours in the image (detected edges)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours), contours  # Return number of contours and the contours