import numpy as np
from sklearn.cluster import KMeans as SKKMeans

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from base_palette_creator import Base_palette_creator

class KMeans(Base_palette_creator):
    """K-means palette creator."""

    def __init__(self, num_colors = 6):
        super().__init__("K-means", num_colors)

    def create_palette(self, images: list[np.ndarray]) -> list[tuple]:
        """Create palette from images.

        Args:
            images (list): List of images.
            num_colors (int, optional): Number of colors in palette. Defaults to 6.

        Returns:
            list[tuple]: List of  a tupple with the palette colors the proportion of pixels in the image for each color.
        """
        num_colors = self.get_num_colors()
        palette = []

        # Convert images to numpy arrays
        images = [image.reshape(-1, image.shape[-1]) for image in images] # Reshape images to only preserve the color channels

        for image in images:
            # Create k-means model
            kmeans = SKKMeans(n_clusters = num_colors, random_state = 0).fit(image)

            # Get colors
            colors = kmeans.cluster_centers_.astype(int)

            # Get proportion of pixels in each color
            proportions = [list(kmeans.labels_).count(i) / len(kmeans.labels_) for i in range(num_colors)]

            # Sort colors and proportions by proportion
            colors, proportions = zip(*sorted(zip(colors, proportions), key = lambda x: x[1], reverse = True))

            palette.append((colors, proportions))

        return palette