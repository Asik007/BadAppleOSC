import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

def plot_colored_edges(image, contours):
    """
    Plots edges with different colors for each continuous object.

    Args:
        image (numpy.ndarray): Input image.
        contours (list): List of contours to be drawn.
    """
    # Create an empty image to draw the contours
    output_image = image.copy()
    # output_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Assign random colors to each contour and draw them
    for contour in contours:
        color = np.random.randint(0, 255, size=3).tolist()  # Random color
        print(f"Contour color: {color}")
        cv2.drawContours(output_image, [contour], -1, color, 3)

    # Plot the result
    plt.figure(figsize=(10, 10))
    # plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.imshow(output_image)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(6)
    plt.close()
