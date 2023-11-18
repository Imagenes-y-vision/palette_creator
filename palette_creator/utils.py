import numpy as np
import matplotlib.pyplot as plt


def show_palette(palette, color_size=10, img=None):
    """
    Display a palette of colors

    Args:
        palette (list): List of colors
        color_size (int): Size of the color square
        img (np.array): Original image to display
    """
    n_subplots = len(palette) + 1 if img is not None else len(palette)
    fig, axs = plt.subplots(1, n_subplots, figsize=(n_subplots * 2, 2))

    # Display the original image
    if img is not None:
        axs[0].imshow(img)
        axs[0].axis("off")

    for i, color in enumerate(palette):
        ax = axs[i] if img is None else axs[i + 1]
        color_img = np.array([[color] * color_size] * color_size).astype(int)
        # Display a colored square
        ax.imshow(color_img)

        # Remove axes and labels
        ax.axis("off")

        # Show the figure
    plt.show()
