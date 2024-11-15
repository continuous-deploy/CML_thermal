import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.image as mpimg
import glob
import os

# Paths to your images
def get_recent_images(metrics_folder_path:str="metrics", num_images=4):
    # Collect all image paths in subfolders
    all_image_paths = glob.glob(os.path.join(metrics_folder_path, '**', '*.png'), recursive=True)

    # Sort images by modification time (latest first)
    all_image_paths.sort(key=os.path.getmtime, reverse=True)

    # Select the most recent images
    recent_image_paths = all_image_paths[:num_images]
    return recent_image_paths


def create_report(image_paths = get_recent_images()):
    # Create a figure and a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Loop through each subplot and each image
    for i, ax in enumerate(axs.flat):
        img = mpimg.imread(image_paths[i])  # Read the image
        ax.imshow(img)                      # Display the image
        ax.axis('off')                      # Hide the axes
        ax.set_title(image_paths[i], fontsize=8, wrap=True)

    # Adjust layout and save the final report image
    plt.tight_layout()
    plt.savefig(f'combined_report{datetime.now().datetime()}.png', dpi=300)  # Save as a high-resolution image
    plt.show()
