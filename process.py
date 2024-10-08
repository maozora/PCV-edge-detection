import os
import cv2
import initialize
import countObjects

def process_image():
    # List all files in the current directory
    files = [f for f in os.listdir('.') if f.endswith(('.jpg', '.jpeg', '.png'))]

    if not files:
        print("No image files found in the current directory.")
        return

    # Display files available in the library (current directory)
    print("Choose a file from the list below:")
    for idx, file in enumerate(files, 1):
        print(f"{idx}. {file}")

    # Ask user to choose the file number
    file_choice = int(input(f"Enter the number of the file you want to use (1-{len(files)}): ")) - 1
    filename = files[file_choice]

    # Load the image in grayscale
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Let the user choose the edge detection method
    print("\nChoose an edge detection method: ")
    print("1. Canny")
    print("2. Sobel")
    print("3. Roberts")

    method_choice = int(input("Enter the number of the method (1-3): "))

    # Map the user choice to the method name
    methods = {1: 'canny', 2: 'sobel', 3: 'roberts'}
    method = methods.get(method_choice, 'canny')  # Default to Canny if input is invalid

    # Step 1: Show the original image
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)

    # Step 2: Edge detection based on user's choice
    edges = initialize.preprocess_and_detect(image, method=method)

    # Step 3: Show the edge detection result
    cv2.imshow('Edge Detection', edges)
    cv2.waitKey(0)

    # Step 4: Count the objects (contours)
    num_objects, contours = countObjects.count_objects(edges)

    # Step 5: Draw contours on the original image
    contour_image = cv2.drawContours(image.copy(), contours, -1, (255, 255, 255), 1)

    # Step 6: Show the contours and object count
    cv2.imshow('Contours', contour_image)
    print(f"\nNumber of objects detected: {num_objects}")

    # Wait for a key press before closing all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()