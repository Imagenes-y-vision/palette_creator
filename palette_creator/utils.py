import numpy as np
import matplotlib.pyplot as plt

def compare_palettes(palettes, names, color_size=10, img=None):
    """
    Compare a palettes of colors

    Args:
        palette (list): List of palettes
        names (list): List of names for each palette
        color_size (int): Size of the color square
        img (np.array): Original image to display
    """
    # Display the original image
    if img is not None:
        plt.imshow(img)
        plt.axis("off")
        
    # n_subplots = len(palettes) + 1 if img is not None else len(palettes)
    n_subplots = len(palettes)
    rows = n_subplots // 4 + 1
    columns = 4 if n_subplots > 4 else n_subplots
    fig, axs = plt.subplots(rows, columns, figsize=(15, 5))
    axs = axs.flatten()


    for i, ax in enumerate(axs):
        if i >= len(palettes):
            ax.axis("off")
            continue
        palette = eval(palettes[i])
        palette_img = np.array([])
        # ax = axs[i]
        for color in palette:
            color_img = np.array([[color] * color_size] * color_size).astype(int)
            # Display a colored square
            palette_img = np.concatenate([palette_img, color_img], axis=1) if palette_img.size else color_img

        ax.set_title(names[i], fontsize=8)
        ax.imshow(palette_img)

        # Remove axes and labels
        ax.axis("off")

        # Show the figure
    plt.show()

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

def get_mse(image: np.ndarray, quantized_image: np.ndarray) -> float:
        """
        Calculate the Mean Squared Error between two images.
        """
        return np.mean((image - quantized_image) ** 2)

def quantize_image(image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    reshaped_image = image.reshape(-1, 1, 3)
    reshaped_palette = palette.reshape(1, -1, 3)

    # Calculate the L2 norm (Euclidean distance) between each pixel and each palette color
    distances = np.linalg.norm(reshaped_image - reshaped_palette, axis=2)

    # Find the index of the nearest color for each pixel
    nearest_color_indices = np.argmin(distances, axis=1)

    # Map the pixels to the nearest colors
    mapped_image = palette[nearest_color_indices].reshape(image.shape)
    return mapped_image
    
def bit_cut(image: np.ndarray, k=3) -> np.ndarray:
    return image//(2**k) + 1

def bit_amplify(image: np.ndarray, k=3) -> np.ndarray:
    return image*(2**k) - 1

def gaussian_filter(image: np.ndarray) -> np.ndarray:
    pass
