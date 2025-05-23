import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import glob

def plot_colored_edges(image, contours, seed=None, pause_time=6):
    """
    Plots edges with different colors for each continuous object.

    Args:
        image (numpy.ndarray): Input image (should be in RGB or already converted).
        contours (list): List of contours to be drawn.
        seed (int, optional): Random seed for reproducibility.
        pause_time (int, optional): Time to display the plot in seconds.
    """
    if seed is not None:
        np.random.seed(seed)

    output_image = image.copy()

    # Generate random colors for all contours at once
    colors = np.random.randint(0, 256, size=(len(contours), 3)).tolist()

    for contour, color in zip(contours, colors):
        cv2.drawContours(output_image, [contour], -1, color, 3)

    plt.figure(figsize=(10, 10))
    plt.imshow(output_image)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(pause_time)
    plt.close()

def save_vid(images, vid_path, fps=30):
    """
    Saves a list of images as a video.

    Args:
        images (list): List of images to be saved.
        vid_path (str): Path to save the video.
        fps (int): Frames per second for the video.
    """
    if not images:
        print("No images to save.")
        return

    height, width, layers = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(vid_path, fourcc, fps, (width, height))

    for img in images:
        out.write(img)
        cv2.imshow('frame', frame)

    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved at {vid_path}")

# Example usage
if __name__ == "__main__":
    # Load an example image
    
    images_path = glob.glob(r'./Test_Vid/Bad Apple (Img)/*.png')
    for img_path in images_path:
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error loading image: {img_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Convert to grayscale and apply Canny edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Plot the colored edges
        plot_colored_edges(image, contours, seed=42, pause_time=.1)